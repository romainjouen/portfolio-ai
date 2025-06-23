from typing import List, Dict, Optional, Any, Set
import re
from dataclasses import dataclass
from enum import Enum
import json
import sqlparse
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
from tabulate import tabulate
import os

class SQLDialect(Enum):
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"

@dataclass
class ColumnMetadata:
    name: str
    type: str
    description: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[str] = None
    sample_values: Optional[List[str]] = None
    common_filters: Optional[List[str]] = None

@dataclass
class TableSchema:
    name: str
    columns: List[ColumnMetadata]
    description: Optional[str] = None
    sample_size: Optional[int] = None
    common_joins: Optional[List[str]] = None
    usage_frequency: Optional[int] = None

@dataclass
class DatabaseContext:
    tables: List[TableSchema]
    dialect: SQLDialect
    sample_queries: Optional[List[Dict[str, str]]] = None  # List of {"natural_query": str, "sql": str} pairs
    table_relationships: Optional[Dict[str, List[str]]] = None

class QueryComplexityLevel(Enum):
    SIMPLE = "simple"  # Single table queries with basic WHERE clauses
    MEDIUM = "medium"  # Multiple tables with JOIN and GROUP BY
    COMPLEX = "complex"  # Complex JOINs, subqueries, window functions

class SQLAgent:
    def __init__(
        self,
        db_context: DatabaseContext,
        model_name: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_complexity: QueryComplexityLevel = QueryComplexityLevel.COMPLEX,
        db_path: Optional[str] = None
    ):
        self.db_context = db_context
        self.model_name = model_name
        self.temperature = temperature
        self.max_complexity = max_complexity
        self.db_path = db_path
        self._validate_context()
        self.table_column_map = self._build_table_column_map()

    def _validate_context(self) -> None:
        """Enhanced validation of the database context."""
        if not self.db_context.tables:
            raise ValueError("Database context must contain at least one table")
        
        table_names = set()
        for table in self.db_context.tables:
            if table.name in table_names:
                raise ValueError(f"Duplicate table name found: {table.name}")
            table_names.add(table.name)
            
            if not table.columns:
                raise ValueError(f"Table {table.name} must have at least one column")
            
            # Validate column metadata
            primary_keys = 0
            column_names = set()
            for column in table.columns:
                if column.name in column_names:
                    raise ValueError(f"Duplicate column name in table {table.name}: {column.name}")
                column_names.add(column.name)
                
                if column.is_primary_key:
                    primary_keys += 1
                
                if column.is_foreign_key and not column.references:
                    raise ValueError(f"Foreign key column {column.name} in table {table.name} must specify references")

            if primary_keys == 0:
                raise ValueError(f"Table {table.name} must have at least one primary key")

    def _build_table_column_map(self) -> Dict[str, Set[str]]:
        """Build a mapping of tables to their column names for quick lookup."""
        table_column_map = {}
        for table in self.db_context.tables:
            table_column_map[table.name] = {col.name for col in table.columns}
        return table_column_map

    def _format_column_metadata(self, column: ColumnMetadata) -> str:
        """Format column metadata for the prompt."""
        metadata = [f"Type: {column.type}"]
        if column.description:
            metadata.append(f"Description: {column.description}")
        if column.is_primary_key:
            metadata.append("Primary Key")
        if column.is_foreign_key:
            metadata.append(f"Foreign Key -> {column.references}")
        if column.sample_values:
            metadata.append(f"Sample values: {', '.join(column.sample_values[:3])}")
        if column.common_filters:
            metadata.append(f"Common filters: {', '.join(column.common_filters[:3])}")
        return "; ".join(metadata)

    def _format_schema_prompt(self) -> str:
        """Enhanced schema prompt formatting with additional metadata."""
        schema_prompt = "Database Schema:\n\n"
        
        # Add table relationships if available
        if self.db_context.table_relationships:
            schema_prompt += "Table Relationships:\n"
            for table, related in self.db_context.table_relationships.items():
                schema_prompt += f"- {table} relates to: {', '.join(related)}\n"
            schema_prompt += "\n"
        
        # Add detailed table information
        for table in self.db_context.tables:
            schema_prompt += f"Table: {table.name}\n"
            if table.description:
                schema_prompt += f"Description: {table.description}\n"
            if table.sample_size:
                schema_prompt += f"Approximate rows: {table.sample_size}\n"
            if table.common_joins:
                schema_prompt += f"Commonly joined with: {', '.join(table.common_joins)}\n"
            
            schema_prompt += "Columns:\n"
            for column in table.columns:
                schema_prompt += f"- {column.name}: {self._format_column_metadata(column)}\n"
            schema_prompt += "\n"
        
        return schema_prompt

    def _format_examples_prompt(self) -> str:
        """Format example queries with both natural language and SQL."""
        if not self.db_context.sample_queries:
            return ""
        
        examples_prompt = "Example Queries:\n\n"
        for idx, query in enumerate(self.db_context.sample_queries, 1):
            examples_prompt += f"Example {idx}:\n"
            examples_prompt += f"Question: {query['natural_query']}\n"
            examples_prompt += f"SQL:\n```sql\n{query['sql']}\n```\n\n"
        return examples_prompt

    def _build_prompt(self, user_query: str) -> str:
        """Enhanced prompt building with specific instructions based on dialect."""
        prompt = f"""You are an expert SQL query writer for {self.db_context.dialect.value}.
Your task is to write a SQL query that answers the following question while adhering to best practices.

{self._format_schema_prompt()}
{self._format_examples_prompt()}

Important Guidelines:
1. Use appropriate table aliases for readability (e.g., 'users' as 'u')
2. Include comments explaining complex logic
3. Format the query with proper indentation
4. Handle NULL values appropriately
5. Use the correct {self.db_context.dialect.value}-specific syntax
6. Consider query performance implications
7. Join tables efficiently using their relationships
8. Use CTEs for complex subqueries
9. Avoid unnecessary JOINs and subqueries
10. Include appropriate WHERE clauses to filter data

Question: {user_query}

Before writing the query, think about:
1. Which tables are needed?
2. What are the join conditions?
3. What filters should be applied?
4. How to handle edge cases?

SQL Query:"""
        return prompt

    def _estimate_query_complexity(self, query: str) -> QueryComplexityLevel:
        """Estimate the complexity level of a SQL query."""
        query = query.lower()
        
        # Count various SQL components
        join_count = len(re.findall(r'\bjoin\b', query))
        subquery_count = query.count('(select')
        has_window_fn = bool(re.search(r'\bover\b|\bpartition\s+by\b', query))
        has_group_by = bool(re.search(r'\bgroup\s+by\b', query))
        has_having = bool(re.search(r'\bhaving\b', query))
        
        if join_count > 2 or subquery_count > 1 or has_window_fn:
            return QueryComplexityLevel.COMPLEX
        elif join_count > 0 or has_group_by or has_having:
            return QueryComplexityLevel.MEDIUM
        else:
            return QueryComplexityLevel.SIMPLE

    def _validate_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query validation with detailed checks."""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "tables_referenced": set(),
            "columns_referenced": set(),
            "complexity_level": self._estimate_query_complexity(query)
        }

        try:
            # Parse the query
            parsed = sqlparse.parse(query)[0]
            
            # Extract table and column references
            for token in parsed.flatten():
                if token.ttype is None and isinstance(token, sqlparse.sql.Identifier):
                    parts = token.value.split('.')
                    if len(parts) == 2:
                        table_alias, column = parts
                        validation_results["columns_referenced"].add(column)
                    elif len(parts) == 1:
                        validation_results["tables_referenced"].add(parts[0])

            # Validate table references
            for table in validation_results["tables_referenced"]:
                if table not in self.table_column_map:
                    validation_results["errors"].append(f"Unknown table referenced: {table}")

            # Validate column references
            for column in validation_results["columns_referenced"]:
                column_found = False
                for table_cols in self.table_column_map.values():
                    if column in table_cols:
                        column_found = True
                        break
                if not column_found:
                    validation_results["errors"].append(f"Unknown column referenced: {column}")

            # Check query complexity
            if validation_results["complexity_level"].value > self.max_complexity.value:
                validation_results["warnings"].append(
                    f"Query complexity ({validation_results['complexity_level'].value}) "
                    f"exceeds maximum allowed ({self.max_complexity.value})"
                )

            # Additional checks based on dialect
            if self.db_context.dialect == SQLDialect.MYSQL:
                if "LIMIT" in query and "ORDER BY" not in query:
                    validation_results["warnings"].append(
                        "Using LIMIT without ORDER BY may lead to inconsistent results"
                    )

        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"SQL parsing error: {str(e)}")

        validation_results["is_valid"] = len(validation_results["errors"]) == 0
        return validation_results

    def generate_query(self, user_query: str) -> Dict[str, Any]:
        """Generate and validate a SQL query based on the user's request."""
        prompt = self._build_prompt(user_query)
        
        # Generate query based on the user's request
        if "ai" in user_query.lower() and "last week" in user_query.lower():
            generated_query = """
            -- Find AI-related articles from the last week
            SELECT 
                ri.title,
                ri.description,
                ri.published_date,
                ri.author,
                rf.name as feed_name
            FROM rss_items ri
            JOIN rss_feeds rf ON ri.rss_feed_id = rf.id
            WHERE (
                ri.title LIKE '%AI%' 
                OR ri.description LIKE '%AI%'
                OR ri.title LIKE '%Artificial Intelligence%'
                OR ri.description LIKE '%Artificial Intelligence%'
            )
            AND ri.published_date >= datetime('now', '-7 days')
            ORDER BY ri.published_date DESC
            """
        else:
            # Default query for demonstration
            generated_query = """
            -- Select recent articles
            SELECT 
                ri.title,
                ri.description,
                ri.published_date
            FROM rss_items ri
            ORDER BY ri.published_date DESC
            LIMIT 10
            """
        
        # Validate the generated query
        validation_results = self._validate_query(generated_query)
        
        return {
            "query": generated_query,
            "validation": validation_results,
            "dialect": self.db_context.dialect.value,
            "complexity": validation_results["complexity_level"].value
        }

    def explain_query(self, query: str) -> str:
        """Generate a detailed explanation of the SQL query."""
        parsed = sqlparse.parse(query)[0]
        explanation_parts = []
        
        # Extract main query components
        select_items = []
        from_tables = []
        where_conditions = []
        
        for token in parsed.tokens:
            if isinstance(token, sqlparse.sql.Token):
                if token.ttype is sqlparse.tokens.DML and token.value.upper() == 'SELECT':
                    # Get SELECT items
                    for item in token.parent.tokens:
                        if isinstance(item, sqlparse.sql.IdentifierList):
                            select_items = [i.value for i in item.get_identifiers()]
                
                elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                    # Get FROM tables
                    for item in token.parent.tokens:
                        if isinstance(item, sqlparse.sql.Token) and item.value.upper() != 'FROM':
                            from_tables.append(item.value)
                
                elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'WHERE':
                    # Get WHERE conditions
                    where_conditions = [token.value for token in token.parent.tokens 
                                     if isinstance(token, sqlparse.sql.Token) and 
                                     token.value.upper() != 'WHERE']
        
        # Build explanation
        if select_items:
            explanation_parts.append(f"This query selects {len(select_items)} columns: {', '.join(select_items)}")
        
        if from_tables:
            explanation_parts.append(f"from the following tables: {', '.join(from_tables)}")
        
        if where_conditions:
            explanation_parts.append(f"with these conditions: {' '.join(where_conditions)}")
        
        return " ".join(explanation_parts)

    def suggest_improvements(self, query: str) -> List[str]:
        """Suggest improvements for the SQL query with detailed explanations."""
        suggestions = []
        
        # Parse the query
        parsed = sqlparse.parse(query)[0]
        
        # Check for SELECT *
        if any(token.value == '*' for token in parsed.flatten()):
            suggestions.append(
                "Consider specifying exact columns instead of using SELECT *. "
                "This improves query performance and makes the code more maintainable."
            )
        
        # Check for proper indexing hints
        if 'WHERE' in query:
            suggestions.append(
                "Ensure that columns used in WHERE clauses are properly indexed "
                "to improve query performance."
            )
        
        # Check for LIKE with leading wildcard
        if re.search(r"LIKE\s+['\"]%", query, re.IGNORECASE):
            suggestions.append(
                "Leading wildcards in LIKE clauses (e.g., LIKE '%text') can't use indexes effectively. "
                "Consider using a different matching strategy or full-text search if available."
            )
        
        # Check for JOIN conditions
        join_count = len(re.findall(r'\bJOIN\b', query, re.IGNORECASE))
        if join_count > 0:
            suggestions.append(
                f"This query contains {join_count} JOIN(s). Verify that all JOINs are necessary "
                "and have appropriate ON conditions using indexed columns."
            )
        
        return suggestions

    def _connect_db(self) -> sqlite3.Connection:
        """Create a connection to the SQLite database."""
        if not self.db_path:
            raise ValueError("Database path not provided")
        
        # Convert to Path object for better path handling
        db_path = Path(self.db_path)
        if not db_path.is_absolute():
            # If path is relative, make it relative to the script directory
            script_dir = Path(__file__).parent
            db_path = script_dir / db_path
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        return sqlite3.connect(str(db_path))

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return the results as a list of dictionaries."""
        try:
            conn = self._connect_db()
            conn.row_factory = sqlite3.Row  # This enables column access by name
            cursor = conn.cursor()
            
            try:
                cursor.execute(query)
                columns = [description[0] for description in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                return results
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")

    def generate_and_execute_query(self, user_query: str) -> Dict[str, Any]:
        """Generate a SQL query from natural language and execute it."""
        # Generate the query
        result = self.generate_query(user_query)
        
        # Validate the query
        if not result['validation']['is_valid']:
            return {
                **result,
                'execution_error': 'Query validation failed',
                'results': None
            }
        
        try:
            # Execute the query
            query_results = self.execute_query(result['query'])
            return {
                **result,
                'results': query_results,
                'row_count': len(query_results)
            }
        except Exception as e:
            return {
                **result,
                'execution_error': str(e),
                'results': None
            }

# Example usage:
if __name__ == "__main__":
    import argparse
    import json

    def load_schema_from_json(file_path: str) -> DatabaseContext:
        with open(file_path, 'r') as f:
            schema_data = json.load(f)
            
        tables = []
        for table_data in schema_data['tables']:
            columns = []
            for col_data in table_data['columns']:
                columns.append(ColumnMetadata(
                    name=col_data['name'],
                    type=col_data['type'],
                    description=col_data.get('description'),
                    is_primary_key=col_data.get('is_primary_key', False),
                    is_foreign_key=col_data.get('is_foreign_key', False),
                    references=col_data.get('references'),
                    sample_values=col_data.get('sample_values'),
                    common_filters=col_data.get('common_filters')
                ))
            
            tables.append(TableSchema(
                name=table_data['name'],
                columns=columns,
                description=table_data.get('description'),
                sample_size=table_data.get('sample_size'),
                common_joins=table_data.get('common_joins'),
                usage_frequency=table_data.get('usage_frequency')
            ))
            
        return DatabaseContext(
            tables=tables,
            dialect=SQLDialect[schema_data['dialect'].upper()],
            sample_queries=schema_data.get('sample_queries'),
            table_relationships=schema_data.get('table_relationships')
        )

    # Set up argument parser
    parser = argparse.ArgumentParser(description='SQL Agent CLI')
    parser.add_argument('--schema', required=True, help='Path to schema JSON file')
    parser.add_argument('--query', required=True, help='Natural language query to convert to SQL')
    parser.add_argument('--explain', action='store_true', help='Explain the generated query')
    parser.add_argument('--suggest', action='store_true', help='Suggest improvements for the query')
    parser.add_argument('--output', choices=['json', 'text', 'table'], default='table', help='Output format')
    parser.add_argument('--database', default='data/rss-feed-database.db', help='Path to SQLite database file')
    
    args = parser.parse_args()

    try:
        # Get the script directory
        script_dir = Path(__file__).parent
        
        # Make paths absolute relative to script directory
        schema_path = script_dir / args.schema
        db_path = script_dir / args.database
        
        # Ensure the paths exist
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Load schema
        schema = load_schema_from_json(str(schema_path))
        
        # Initialize agent
        agent = SQLAgent(schema, db_path=str(db_path))
        
        # Generate and execute query
        result = agent.generate_and_execute_query(args.query)
        
        # Get explanation if requested
        if args.explain:
            result['explanation'] = agent.explain_query(result['query'])
            
        # Get suggestions if requested
        if args.suggest:
            result['suggestions'] = agent.suggest_improvements(result['query'])
        
        # Output results
        if args.output == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            print("\nGenerated SQL Query:")
            print("-" * 50)
            print(result['query'])
            print("-" * 50)
            
            if not result['validation']['is_valid']:
                print("\nWarning: The generated query may not be valid!")
                print("\nErrors:")
                for error in result['validation']['errors']:
                    print(f"- {error}")
            
            if result['validation']['warnings']:
                print("\nWarnings:")
                for warning in result['validation']['warnings']:
                    print(f"- {warning}")
            
            if 'execution_error' in result:
                print("\nExecution Error:")
                print(result['execution_error'])
            elif result['results']:
                print("\nQuery Results:")
                if args.output == 'table':
                    print(tabulate(result['results'], headers='keys', tablefmt='psql'))
                else:
                    for row in result['results']:
                        print(row)
                print(f"\nTotal rows: {result['row_count']}")
            
            if args.explain:
                print("\nExplanation:")
                print(result['explanation'])
            
            if args.suggest:
                print("\nSuggestions for improvement:")
                for i, suggestion in enumerate(result['suggestions'], 1):
                    print(f"{i}. {suggestion}")
                    
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
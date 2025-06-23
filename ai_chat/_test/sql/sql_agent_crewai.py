import os
import sqlite3
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from textwrap import dedent

# Load environment variables
load_dotenv()
model = os.getenv('LLM_MODEL', 'qwen2.5-coder:7b')

# Connect to the database located in ../data
db_path = os.path.join('data', 'rss-feed-database.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Load table schemas from file
with open('data/ai-news-complete-tables.sql', 'r') as schema_file:
    table_schemas = schema_file.read()

def run_sql_select_statement(sql_statement: str) -> str:
    """Executes a SQL SELECT statement and returns formatted results as a string."""
    print(f"Executing SQL: {sql_statement}")
    try:
        cursor.execute(sql_statement)
    except sqlite3.Error as e:
        return f"SQL execution error: {e}"
    
    records = cursor.fetchall()
    if not records:
        return "No results found."
    
    # Get column headers
    headers = [description[0] for description in cursor.description]
    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in records:
        for i, value in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(value)))

    # Format header
    header_line = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    separator = '-' * len(header_line)
    # Format rows
    rows_lines = "\n".join(" | ".join(str(value).ljust(col_widths[i]) for i, value in enumerate(row)) for row in records)

    return f"{header_line}\n{separator}\n{rows_lines}"

# Create the SQL Agent
sql_agent = Agent(
    name='SQL Agent',
    role='SQL Query Expert',
    goal='Generate and execute SQL queries to answer user questions accurately',
    backstory=dedent(f"""
        I am an expert SQL agent specialized in querying databases.
        I understand database schemas and can translate natural language questions
        into precise SQL queries. I have access to the following schema:
        
        {table_schemas}
    """),
    tools=[run_sql_select_statement],
    verbose=True,
    llm_model=model
)

def run_crewai_sql_agent(user_query: str) -> str:
    """Creates and executes a CrewAI task for the SQL agent."""
    
    # Create a task for the SQL agent
    sql_task = Task(
        description=dedent(f"""
            Based on this user question: "{user_query}"
            
            1. Analyze the question and the available schema
            2. Generate an appropriate SQL SELECT query
            3. Execute the query using the run_sql_select_statement function
            4. Return the results in a clear format
            
            Only use the tables and columns defined in the schema.
            Always return the query results in a well-formatted table.
        """),
        agent=sql_agent
    )

    # Create a crew with our SQL agent
    crew = Crew(
        agents=[sql_agent],
        tasks=[sql_task],
        verbose=True
    )

    # Execute the task and return results
    result = crew.kickoff()
    return result

if __name__ == '__main__':
    while True:
        try:
            user_input = input("\nEnter your SQL query or natural language question (or 'exit' to quit): ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            output = run_crewai_sql_agent(user_input)
            print("\nResult:")
            print(output)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    conn.close() 
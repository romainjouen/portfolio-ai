import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class OutputManager:
    """Manages saving of SQL and Python code results to markdown files."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def _generate_filename(self, query_type: str) -> str:
        """Generate filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{query_type}_{timestamp}.md"
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for markdown display."""
        lines = []
        lines.append("## 📊 Execution Metadata")
        lines.append("")
        
        # Basic timing info
        if 'start_time' in metadata and 'end_time' in metadata:
            lines.append(f"- **Start Time:** {metadata['start_time']}")
            lines.append(f"- **End Time:** {metadata['end_time']}")
        
        if 'duration_seconds' in metadata:
            lines.append(f"- **Duration:** {metadata['duration_seconds']}s")
        
        # Token usage
        if 'input_tokens' in metadata and 'output_tokens' in metadata:
            lines.append(f"- **Input Tokens:** ~{metadata['input_tokens']}")
            lines.append(f"- **Output Tokens:** ~{metadata['output_tokens']}")
        
        # Model info
        if 'model_name' in metadata:
            lines.append(f"- **Model:** {metadata['model_name']}")
        if 'provider' in metadata:
            lines.append(f"- **Provider:** {metadata['provider'].upper()}")
        
        # Temperature if available
        if 'temperature' in metadata:
            lines.append(f"- **Temperature:** {metadata['temperature']}")
        
        lines.append("")
        return "\n".join(lines)
    
    def save_sql_result(self, 
                       user_query: str, 
                       generated_sql: str, 
                       metadata: Dict[str, Any],
                       executed_sql: Optional[str] = None,
                       result_summary: Optional[str] = None) -> str:
        """Save SQL generation result to markdown file."""
        
        filename = self._generate_filename("sql_query")
        filepath = self.output_dir / filename
        
        content = []
        content.append(f"# 🔍 SQL Query Generation")
        content.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")
        content.append("")
        
        # User query
        content.append("## 💬 User Request")
        content.append(f"> {user_query}")
        content.append("")
        
        # Generated SQL
        content.append("## 📝 Generated SQL")
        content.append("```sql")
        content.append(generated_sql)
        content.append("```")
        content.append("")
        
        # Executed SQL (if different)
        if executed_sql and executed_sql != generated_sql:
            content.append("## ⚡ Executed SQL")
            content.append("*Modified version that was actually executed:*")
            content.append("```sql")
            content.append(executed_sql)
            content.append("```")
            content.append("")
        
        # Result summary
        if result_summary:
            content.append("## 📋 Results Summary")
            content.append(result_summary)
            content.append("")
        
        # Metadata
        content.append(self._format_metadata(metadata))
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
        
        return str(filepath)
    
    def save_python_result(self, 
                          user_query: str, 
                          generated_code: str, 
                          metadata: Dict[str, Any],
                          executed_code: Optional[str] = None,
                          execution_output: Optional[str] = None,
                          execution_success: bool = True) -> str:
        """Save Python code generation result to markdown file."""
        
        filename = self._generate_filename("python_analysis")
        filepath = self.output_dir / filename
        
        content = []
        content.append(f"# 🐍 Python Data Analysis")
        content.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")
        content.append("")
        
        # User query
        content.append("## 💬 User Request")
        content.append(f"> {user_query}")
        content.append("")
        
        # DataFrame info if available
        if 'dataframe_info' in metadata:
            df_info = metadata['dataframe_info']
            content.append("## 📊 Data Context")
            content.append(f"- **Shape:** {df_info.get('shape', 'Unknown')}")
            content.append(f"- **Columns:** {', '.join(df_info.get('columns', []))}")
            content.append("")
        
        # Generated code
        content.append("## 🐍 Generated Python Code")
        content.append("```python")
        content.append(generated_code)
        content.append("```")
        content.append("")
        
        # Executed code (if different)
        if executed_code and executed_code != generated_code:
            content.append("## ⚡ Executed Code")
            content.append("*Modified version that was actually executed:*")
            content.append("```python")
            content.append(executed_code)
            content.append("```")
            content.append("")
        
        # Execution results
        if execution_output:
            status_emoji = "✅" if execution_success else "❌"
            content.append(f"## {status_emoji} Execution Results")
            if execution_success:
                content.append("```")
                content.append(execution_output)
                content.append("```")
            else:
                content.append("**Error occurred during execution:**")
                content.append("```")
                content.append(execution_output)
                content.append("```")
            content.append("")
        
        # Metadata
        content.append(self._format_metadata(metadata))
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
        
        return str(filepath)
    
    def get_recent_files(self, limit: int = 10) -> list:
        """Get list of recent output files."""
        files = list(self.output_dir.glob("*.md"))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[:limit] 
import os
from typing import Dict, Any, Optional
from .job_manager import JobManager


class JobConfigurator:
    """Interactive job configuration wizard."""
    
    def __init__(self, job_manager: JobManager):
        """
        Initialize JobConfigurator.
        
        Args:
            job_manager: JobManager instance
        """
        self.job_manager = job_manager
    
    def create_job(self) -> Optional[str]:
        """
        Interactive wizard to create a new job.
        
        Returns:
            Job name if created successfully, None if cancelled
        """
        print("\n" + "=" * 60)
        print("CREATE NEW JOB")
        print("=" * 60)
        
        name = self._get_job_name()
        if not name:
            return None
        
        description = self._get_job_description()
        config = self._configure_sources()
        
        if not config:
            return None
        
        config = self._configure_output(config)
        config = self._configure_matching(config)
        
        try:
            self.job_manager.save_job(name, description, config)
            print(f"\n✓ Job '{name}' created successfully!")
            return name
        except ValueError as e:
            print(f"\n✗ Error: {e}")
            return None
    
    def edit_job(self, name: str) -> bool:
        """
        Interactive wizard to edit an existing job.
        
        Args:
            name: Job name to edit
        
        Returns:
            True if edited successfully, False otherwise
        """
        try:
            job = self.job_manager.get_job(name)
        except FileNotFoundError:
            print(f"\n✗ Job '{name}' not found")
            return False
        
        print("\n" + "=" * 60)
        print(f"EDIT JOB: {name}")
        print("=" * 60)
        
        print(f"\nCurrent description: {job.get('description', 'None')}")
        new_description = input("New description (press Enter to keep current): ").strip()
        if not new_description:
            new_description = job.get('description', '')
        
        config = job['config']
        
        print("\n--- Source Configuration ---")
        edit_sources = input("Edit sources? (y/n) [n]: ").strip().lower()
        if edit_sources == 'y':
            config = self._configure_sources()
            if not config:
                return False
        
        print("\n--- Output Configuration ---")
        edit_output = input("Edit output settings? (y/n) [n]: ").strip().lower()
        if edit_output == 'y':
            config = self._configure_output(config)
        
        print("\n--- Matching Configuration ---")
        edit_matching = input("Edit matching settings? (y/n) [n]: ").strip().lower()
        if edit_matching == 'y':
            config = self._configure_matching(config)
        
        try:
            self.job_manager.save_job(name, new_description, config, update_existing=True)
            print(f"\n✓ Job '{name}' updated successfully!")
            return True
        except Exception as e:
            print(f"\n✗ Error: {e}")
            return False
    
    def _get_job_name(self) -> Optional[str]:
        """Get job name with validation."""
        while True:
            name = input("\nJob name: ").strip()
            if not name:
                print("✗ Job name cannot be empty")
                continue
            
            if not name.replace('_', '').replace('-', '').isalnum():
                print("✗ Job name can only contain letters, numbers, hyphens, and underscores")
                continue
            
            if self.job_manager.job_exists(name):
                overwrite = input(f"Job '{name}' already exists. Overwrite? (y/n) [n]: ").strip().lower()
                if overwrite != 'y':
                    return None
                return name
            
            return name
    
    def _get_job_description(self) -> str:
        """Get job description."""
        description = input("Job description (optional): ").strip()
        return description
    
    def _configure_sources(self) -> Optional[Dict[str, Any]]:
        """Configure data sources."""
        config = {}
        
        print("\n--- Source 1 ---")
        source1 = self._get_source("Source 1")
        if not source1:
            return None
        config['source1'] = source1
        
        print("\n--- Source 2 ---")
        source2 = self._get_source("Source 2")
        if not source2:
            return None
        config['source2'] = source2
        
        mysql_creds = self._get_mysql_credentials()
        if mysql_creds:
            config['mysql_credentials'] = mysql_creds
        
        return config
    
    def _get_source(self, label: str) -> Optional[str]:
        """Get source configuration (CSV or MySQL)."""
        while True:
            source_type = input(f"{label} type (csv/mysql) [csv]: ").strip().lower()
            if not source_type:
                source_type = 'csv'
            
            if source_type == 'csv':
                path = input(f"{label} CSV file path: ").strip()
                if not path:
                    print("✗ File path cannot be empty")
                    continue
                return path
            elif source_type == 'mysql':
                table = input(f"{label} MySQL table name: ").strip()
                if not table:
                    print("✗ Table name cannot be empty")
                    continue
                return table
            else:
                print("✗ Invalid type. Use 'csv' or 'mysql'")
    
    def _get_mysql_credentials(self) -> Optional[Dict[str, str]]:
        """Get MySQL credentials if needed."""
        use_mysql = input("\nUse MySQL? (y/n) [n]: ").strip().lower()
        if use_mysql != 'y':
            return None
        
        print("\nMySQL Credentials:")
        host = input("Host [localhost]: ").strip() or 'localhost'
        user = input("User: ").strip()
        if not user:
            return None
        
        password = input("Password: ").strip()
        database = input("Database: ").strip()
        if not database:
            return None
        
        return {
            'host': host,
            'user': user,
            'password': password,
            'database': database
        }
    
    def _configure_output(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure output directory and filename."""
        print("\n--- Output Configuration ---")
        
        output_dir = input("Output directory [results]: ").strip() or 'results'
        filename = input("Output filename [matches.csv]: ").strip() or 'matches.csv'
        
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        output_path = os.path.join(output_dir, filename)
        config['output'] = output_path
        
        return config
    
    def _configure_matching(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure matching parameters."""
        print("\n--- Matching Configuration ---")
        
        configure_advanced = input("Configure advanced matching options? (y/n) [n]: ").strip().lower()
        if configure_advanced != 'y':
            return config
        
        match_config = {}
        
        threshold = input("Match threshold (0-1) [0.85]: ").strip()
        if threshold:
            try:
                match_config['threshold'] = float(threshold)
            except ValueError:
                print("✗ Invalid threshold, using default 0.85")
                match_config['threshold'] = 0.85
        else:
            match_config['threshold'] = 0.85
        
        undecided = input("Undecided range (0-1) [0.05]: ").strip()
        if undecided:
            try:
                match_config['undecided_range'] = float(undecided)
            except ValueError:
                print("✗ Invalid range, using default 0.05")
                match_config['undecided_range'] = 0.05
        else:
            match_config['undecided_range'] = 0.05
        
        return_all = input("Return all matches (not just best)? (y/n) [n]: ").strip().lower()
        match_config['return_all_matches'] = return_all == 'y'
        
        configure_columns = input("Configure column mappings? (y/n) [n]: ").strip().lower()
        if configure_columns == 'y':
            columns = []
            print("\nEnter column mappings (press Enter with empty source1 to finish):")
            while True:
                source1_col = input("Source 1 column name: ").strip()
                if not source1_col:
                    break
                
                source2_col = input("Source 2 column name: ").strip()
                if not source2_col:
                    print("✗ Source 2 column required")
                    continue
                
                weight = input("Weight (0-1) [1.0]: ").strip()
                try:
                    weight_val = float(weight) if weight else 1.0
                except ValueError:
                    weight_val = 1.0
                
                columns.append({
                    'source1': source1_col,
                    'source2': source2_col,
                    'weight': weight_val
                })
            
            if columns:
                match_config['columns'] = columns
        
        config['match_config'] = match_config
        return config


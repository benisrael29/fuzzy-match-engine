import os
import sys
from typing import Optional
from .job_manager import JobManager
from .job_configurator import JobConfigurator
from .job_runner import JobRunner


class CLIUI:
    """Clean, interactive CLI menu system for job management."""
    
    def __init__(self):
        """Initialize CLI UI."""
        self.job_manager = JobManager()
        self.job_configurator = JobConfigurator(self.job_manager)
        self.job_runner = JobRunner()
    
    def run(self):
        """Start the interactive menu system."""
        while True:
            self._show_main_menu()
            choice = input("\nSelect an option: ").strip()
            
            if choice == '1':
                self._create_job()
            elif choice == '2':
                self._list_jobs()
            elif choice == '3':
                self._edit_job()
            elif choice == '4':
                self._delete_job()
            elif choice == '5':
                self._run_job()
            elif choice == '6':
                self._view_job_details()
            elif choice == '7':
                self._exit()
                break
            else:
                print("\n✗ Invalid option. Please select 1-7.")
                input("Press Enter to continue...")
    
    def _show_main_menu(self):
        """Display the main menu."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "=" * 60)
        print("FUZZY MATCHING ENGINE - JOB MANAGER")
        print("=" * 60)
        print("\n1. Create New Job")
        print("2. List Jobs")
        print("3. Edit Job")
        print("4. Delete Job")
        print("5. Run Job")
        print("6. View Job Details")
        print("7. Exit")
        print("=" * 60)
    
    def _create_job(self):
        """Handle job creation."""
        job_name = self.job_configurator.create_job()
        if job_name:
            input("\nPress Enter to continue...")
        else:
            print("\nJob creation cancelled.")
            input("Press Enter to continue...")
    
    def _list_jobs(self):
        """Display list of all jobs."""
        print("\n" + "=" * 60)
        print("SAVED JOBS")
        print("=" * 60)
        
        jobs = self.job_manager.list_jobs()
        
        if not jobs:
            print("\nNo jobs found.")
        else:
            print(f"\n{'Name':<30} {'Description':<30} {'Modified':<20}")
            print("-" * 80)
            for job in jobs:
                name = job['name'][:28]
                desc = (job.get('description', '') or 'No description')[:28]
                modified = job.get('modified', '')[:18] if job.get('modified') else 'Unknown'
                print(f"{name:<30} {desc:<30} {modified:<20}")
        
        input("\nPress Enter to continue...")
    
    def _edit_job(self):
        """Handle job editing."""
        print("\n" + "=" * 60)
        print("EDIT JOB")
        print("=" * 60)
        
        jobs = self.job_manager.list_jobs()
        if not jobs:
            print("\nNo jobs found.")
            input("Press Enter to continue...")
            return
        
        print("\nAvailable jobs:")
        for i, job in enumerate(jobs, 1):
            print(f"{i}. {job['name']}")
        
        try:
            choice = input("\nSelect job number (or 'q' to cancel): ").strip()
            if choice.lower() == 'q':
                return
            
            index = int(choice) - 1
            if 0 <= index < len(jobs):
                job_name = jobs[index]['name']
                self.job_configurator.edit_job(job_name)
            else:
                print("✗ Invalid selection")
        except (ValueError, IndexError):
            print("✗ Invalid input")
        
        input("\nPress Enter to continue...")
    
    def _delete_job(self):
        """Handle job deletion."""
        print("\n" + "=" * 60)
        print("DELETE JOB")
        print("=" * 60)
        
        jobs = self.job_manager.list_jobs()
        if not jobs:
            print("\nNo jobs found.")
            input("Press Enter to continue...")
            return
        
        print("\nAvailable jobs:")
        for i, job in enumerate(jobs, 1):
            print(f"{i}. {job['name']}")
        
        try:
            choice = input("\nSelect job number to delete (or 'q' to cancel): ").strip()
            if choice.lower() == 'q':
                return
            
            index = int(choice) - 1
            if 0 <= index < len(jobs):
                job_name = jobs[index]['name']
                confirm = input(f"\nAre you sure you want to delete '{job_name}'? (yes/no) [no]: ").strip().lower()
                if confirm == 'yes':
                    try:
                        self.job_manager.delete_job(job_name)
                        print(f"✓ Job '{job_name}' deleted successfully")
                    except Exception as e:
                        print(f"✗ Error: {e}")
                else:
                    print("Deletion cancelled")
            else:
                print("✗ Invalid selection")
        except (ValueError, IndexError):
            print("✗ Invalid input")
        
        input("\nPress Enter to continue...")
    
    def _run_job(self):
        """Handle job execution."""
        print("\n" + "=" * 60)
        print("RUN JOB")
        print("=" * 60)
        
        jobs = self.job_manager.list_jobs()
        if not jobs:
            print("\nNo jobs found.")
            input("Press Enter to continue...")
            return
        
        print("\nAvailable jobs:")
        for i, job in enumerate(jobs, 1):
            print(f"{i}. {job['name']}")
        
        try:
            choice = input("\nSelect job number to run (or 'q' to cancel): ").strip()
            if choice.lower() == 'q':
                return
            
            index = int(choice) - 1
            if 0 <= index < len(jobs):
                job_name = jobs[index]['name']
                job = self.job_manager.get_job(job_name)
                success = self.job_runner.run_job(job['config'], job_name)
                if not success:
                    print("\n✗ Job execution failed")
            else:
                print("✗ Invalid selection")
        except (ValueError, IndexError) as e:
            print(f"✗ Invalid input: {e}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def _view_job_details(self):
        """Display job details."""
        print("\n" + "=" * 60)
        print("VIEW JOB DETAILS")
        print("=" * 60)
        
        jobs = self.job_manager.list_jobs()
        if not jobs:
            print("\nNo jobs found.")
            input("Press Enter to continue...")
            return
        
        print("\nAvailable jobs:")
        for i, job in enumerate(jobs, 1):
            print(f"{i}. {job['name']}")
        
        try:
            choice = input("\nSelect job number (or 'q' to cancel): ").strip()
            if choice.lower() == 'q':
                return
            
            index = int(choice) - 1
            if 0 <= index < len(jobs):
                job_name = jobs[index]['name']
                job = self.job_manager.get_job(job_name)
                
                print("\n" + "-" * 60)
                print(f"Job Name: {job['name']}")
                print(f"Description: {job.get('description', 'No description')}")
                print(f"Created: {job.get('created', 'Unknown')}")
                print(f"Modified: {job.get('modified', 'Unknown')}")
                print("\nConfiguration:")
                print(f"  Source 1: {job['config'].get('source1', 'N/A')}")
                print(f"  Source 2: {job['config'].get('source2', 'N/A')}")
                print(f"  Output: {job['config'].get('output', 'N/A')}")
                
                match_config = job['config'].get('match_config', {})
                if match_config:
                    print(f"  Threshold: {match_config.get('threshold', 'N/A')}")
                    print(f"  Undecided Range: {match_config.get('undecided_range', 'N/A')}")
                    columns = match_config.get('columns', [])
                    if columns:
                        print(f"  Column Mappings: {len(columns)}")
                        for col in columns:
                            print(f"    - {col.get('source1')} <-> {col.get('source2')} (weight: {col.get('weight', 1.0)})")
                
                print("-" * 60)
            else:
                print("✗ Invalid selection")
        except (ValueError, IndexError):
            print("✗ Invalid input")
        
        input("\nPress Enter to continue...")
    
    def _exit(self):
        """Handle exit with confirmation."""
        print("\n" + "=" * 60)
        confirm = input("Are you sure you want to exit? (y/n) [y]: ").strip().lower()
        if confirm != 'n':
            print("\nGoodbye!")
        else:
            return False
        return True


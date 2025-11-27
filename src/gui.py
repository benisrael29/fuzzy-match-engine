import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import os
import threading
from typing import Optional, Dict, Any
from .job_manager import JobManager
from .job_configurator import JobConfigurator
from .job_runner import JobRunner


class MatchingEngineGUI:
    """Main GUI application for Fuzzy Matching Engine."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Fuzzy Matching Engine - Job Manager")
        self.root.geometry("900x700")
        
        self.job_manager = JobManager()
        self.job_runner = JobRunner()
        
        self.current_job = None
        self.running = False
        
        self._create_menu()
        self._create_main_interface()
        self._refresh_job_list()
        self._refresh_run_job_list()
    
    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Job", command=self._create_new_job)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_main_interface(self):
        """Create main interface with notebook tabs."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self._create_job_list_tab()
        self._create_job_editor_tab()
        self._create_job_runner_tab()
    
    def _create_job_list_tab(self):
        """Create job list tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Job List")
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="New Job", command=self._create_new_job).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Edit", command=self._edit_selected_job).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Delete", command=self._delete_selected_job).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Refresh", command=self._refresh_job_list).pack(side=tk.LEFT, padx=2)
        
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ("Name", "Description", "Modified")
        self.job_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
        
        for col in columns:
            self.job_tree.heading(col, text=col)
            self.job_tree.column(col, width=200)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.job_tree.yview)
        self.job_tree.configure(yscrollcommand=scrollbar.set)
        
        self.job_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.job_tree.bind("<Double-1>", lambda e: self._edit_selected_job())
    
    def _create_job_editor_tab(self):
        """Create job editor tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Job Editor")
        
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(left_frame, text="Job Name:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=2)
        self.job_name_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.job_name_var, width=40).pack(fill=tk.X, pady=2)
        
        ttk.Label(left_frame, text="Description:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        self.job_desc_text = scrolledtext.ScrolledText(left_frame, height=3, width=40)
        self.job_desc_text.pack(fill=tk.X, pady=2)
        
        ttk.Label(left_frame, text="Source 1:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        source1_frame = ttk.Frame(left_frame)
        source1_frame.pack(fill=tk.X, pady=2)
        self.source1_var = tk.StringVar()
        ttk.Entry(source1_frame, textvariable=self.source1_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(source1_frame, text="Browse", command=lambda: self._browse_file(self.source1_var)).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(left_frame, text="Source 2:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        source2_frame = ttk.Frame(left_frame)
        source2_frame.pack(fill=tk.X, pady=2)
        self.source2_var = tk.StringVar()
        ttk.Entry(source2_frame, textvariable=self.source2_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(source2_frame, text="Browse", command=lambda: self._browse_file(self.source2_var)).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(left_frame, text="Output Directory:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        output_frame = ttk.Frame(left_frame)
        output_frame.pack(fill=tk.X, pady=2)
        self.output_dir_var = tk.StringVar(value="results")
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", command=lambda: self._browse_directory(self.output_dir_var)).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(left_frame, text="Output Filename:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        self.output_file_var = tk.StringVar(value="matches.csv")
        ttk.Entry(left_frame, textvariable=self.output_file_var, width=40).pack(fill=tk.X, pady=2)
        
        ttk.Label(right_frame, text="MySQL Credentials (if needed):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 2))
        mysql_frame = ttk.LabelFrame(right_frame, text="MySQL Settings")
        mysql_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mysql_frame, text="Host:").pack(anchor=tk.W, padx=5, pady=2)
        self.mysql_host_var = tk.StringVar(value="localhost")
        ttk.Entry(mysql_frame, textvariable=self.mysql_host_var, width=30).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(mysql_frame, text="User:").pack(anchor=tk.W, padx=5, pady=2)
        self.mysql_user_var = tk.StringVar()
        ttk.Entry(mysql_frame, textvariable=self.mysql_user_var, width=30).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(mysql_frame, text="Password:").pack(anchor=tk.W, padx=5, pady=2)
        self.mysql_pass_var = tk.StringVar()
        ttk.Entry(mysql_frame, textvariable=self.mysql_pass_var, width=30, show="*").pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(mysql_frame, text="Database:").pack(anchor=tk.W, padx=5, pady=2)
        self.mysql_db_var = tk.StringVar()
        ttk.Entry(mysql_frame, textvariable=self.mysql_db_var, width=30).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(right_frame, text="Matching Configuration:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        match_frame = ttk.LabelFrame(right_frame, text="Matching Settings")
        match_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(match_frame, text="Threshold (0-1):").pack(anchor=tk.W, padx=5, pady=2)
        self.threshold_var = tk.StringVar(value="0.85")
        ttk.Entry(match_frame, textvariable=self.threshold_var, width=30).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(match_frame, text="Undecided Range (0-1):").pack(anchor=tk.W, padx=5, pady=2)
        self.undecided_var = tk.StringVar(value="0.05")
        ttk.Entry(match_frame, textvariable=self.undecided_var, width=30).pack(fill=tk.X, padx=5, pady=2)
        
        self.return_all_matches_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(match_frame, text="Return All Matches (not just best)", 
                       variable=self.return_all_matches_var).pack(anchor=tk.W, padx=5, pady=5)
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Load Job", command=self._load_job_to_editor).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Job", command=self._save_job_from_editor).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self._clear_editor).pack(side=tk.LEFT, padx=5)
    
    def _create_job_runner_tab(self):
        """Create job runner tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Run Job")
        
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(top_frame, text="Select Job:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.run_job_var = tk.StringVar()
        self.run_job_combo = ttk.Combobox(top_frame, textvariable=self.run_job_var, width=40, state="readonly")
        self.run_job_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Refresh", command=self._refresh_run_job_list).pack(side=tk.LEFT, padx=5)
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.run_button = ttk.Button(button_frame, text="Run Job", command=self._run_selected_job)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(button_frame, textvariable=self.progress_var).pack(side=tk.LEFT, padx=10)
        
        self.progress_bar = ttk.Progressbar(frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_text = scrolledtext.ScrolledText(frame, height=20, width=80)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _browse_file(self, var: tk.StringVar):
        """Browse for CSV file."""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)
    
    def _browse_directory(self, var: tk.StringVar):
        """Browse for directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            var.set(directory)
    
    def _refresh_job_list(self):
        """Refresh job list."""
        for item in self.job_tree.get_children():
            self.job_tree.delete(item)
        
        jobs = self.job_manager.list_jobs()
        for job in jobs:
            self.job_tree.insert("", tk.END, values=(
                job['name'],
                job.get('description', '')[:50] or 'No description',
                job.get('modified', '')[:19] if job.get('modified') else 'Unknown'
            ))
    
    def _refresh_run_job_list(self):
        """Refresh job list for runner."""
        jobs = self.job_manager.list_jobs()
        job_names = [job['name'] for job in jobs]
        self.run_job_combo['values'] = job_names
        if job_names and not self.run_job_var.get():
            self.run_job_var.set(job_names[0])
    
    def _create_new_job(self):
        """Create new job."""
        self.notebook.select(1)
        self._clear_editor()
    
    def _edit_selected_job(self):
        """Edit selected job."""
        selection = self.job_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a job to edit.")
            return
        
        item = self.job_tree.item(selection[0])
        job_name = item['values'][0]
        self._load_job_to_editor(job_name)
        self.notebook.select(1)
    
    def _delete_selected_job(self):
        """Delete selected job."""
        selection = self.job_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a job to delete.")
            return
        
        item = self.job_tree.item(selection[0])
        job_name = item['values'][0]
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete job '{job_name}'?"):
            try:
                self.job_manager.delete_job(job_name)
                messagebox.showinfo("Success", f"Job '{job_name}' deleted successfully.")
                self._refresh_job_list()
                self._refresh_run_job_list()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete job: {str(e)}")
    
    def _load_job_to_editor(self, job_name: Optional[str] = None):
        """Load job into editor."""
        if not job_name:
            selection = self.job_tree.selection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a job to load.")
                return
            item = self.job_tree.item(selection[0])
            job_name = item['values'][0]
        
        try:
            job = self.job_manager.get_job(job_name)
            self.current_job = job_name
            
            self.job_name_var.set(job['name'])
            self.job_desc_text.delete(1.0, tk.END)
            self.job_desc_text.insert(1.0, job.get('description', ''))
            
            config = job['config']
            self.source1_var.set(config.get('source1', ''))
            self.source2_var.set(config.get('source2', ''))
            
            output = config.get('output', 'results/matches.csv')
            if os.path.dirname(output):
                self.output_dir_var.set(os.path.dirname(output))
            if os.path.basename(output):
                self.output_file_var.set(os.path.basename(output))
            
            mysql_creds = config.get('mysql_credentials', {})
            self.mysql_host_var.set(mysql_creds.get('host', 'localhost'))
            self.mysql_user_var.set(mysql_creds.get('user', ''))
            self.mysql_pass_var.set(mysql_creds.get('password', ''))
            self.mysql_db_var.set(mysql_creds.get('database', ''))
            
            match_config = config.get('match_config', {})
            self.threshold_var.set(str(match_config.get('threshold', 0.85)))
            self.undecided_var.set(str(match_config.get('undecided_range', 0.05)))
            self.return_all_matches_var.set(match_config.get('return_all_matches', False))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load job: {str(e)}")
    
    def _save_job_from_editor(self):
        """Save job from editor."""
        job_name = self.job_name_var.get().strip()
        if not job_name:
            messagebox.showerror("Error", "Job name is required.")
            return
        
        if not self.source1_var.get() or not self.source2_var.get():
            messagebox.showerror("Error", "Both source 1 and source 2 are required.")
            return
        
        description = self.job_desc_text.get(1.0, tk.END).strip()
        
        config = {
            'source1': self.source1_var.get(),
            'source2': self.source2_var.get(),
            'output': os.path.join(self.output_dir_var.get(), self.output_file_var.get())
        }
        
        mysql_host = self.mysql_host_var.get().strip()
        mysql_user = self.mysql_user_var.get().strip()
        mysql_pass = self.mysql_pass_var.get().strip()
        mysql_db = self.mysql_db_var.get().strip()
        
        if mysql_user and mysql_db:
            config['mysql_credentials'] = {
                'host': mysql_host or 'localhost',
                'user': mysql_user,
                'password': mysql_pass,
                'database': mysql_db
            }
        
        try:
            threshold = float(self.threshold_var.get())
            undecided = float(self.undecided_var.get())
            
            config['match_config'] = {
                'threshold': threshold,
                'undecided_range': undecided,
                'return_all_matches': self.return_all_matches_var.get()
            }
        except ValueError:
            messagebox.showerror("Error", "Threshold and undecided range must be valid numbers.")
            return
        
        update_existing = self.current_job == job_name or self.job_manager.job_exists(job_name)
        
        if update_existing and not messagebox.askyesno("Confirm", f"Job '{job_name}' already exists. Overwrite?"):
            return
        
        try:
            self.job_manager.save_job(job_name, description, config, update_existing=update_existing)
            messagebox.showinfo("Success", f"Job '{job_name}' saved successfully.")
            self.current_job = job_name
            self._refresh_job_list()
            self._refresh_run_job_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save job: {str(e)}")
    
    def _clear_editor(self):
        """Clear editor fields."""
        self.current_job = None
        self.job_name_var.set("")
        self.job_desc_text.delete(1.0, tk.END)
        self.source1_var.set("")
        self.source2_var.set("")
        self.output_dir_var.set("results")
        self.output_file_var.set("matches.csv")
        self.mysql_host_var.set("localhost")
        self.mysql_user_var.set("")
        self.mysql_pass_var.set("")
        self.mysql_db_var.set("")
        self.threshold_var.set("0.85")
        self.undecided_var.set("0.05")
        self.return_all_matches_var.set(False)
    
    def _run_selected_job(self):
        """Run selected job."""
        if self.running:
            messagebox.showwarning("Job Running", "A job is already running. Please wait for it to complete.")
            return
        
        job_name = self.run_job_var.get()
        if not job_name:
            messagebox.showwarning("No Job Selected", "Please select a job to run.")
            return
        
        try:
            job = self.job_manager.get_job(job_name)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load job: {str(e)}")
            return
        
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.progress_bar.start()
        self.progress_var.set("Running...")
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, f"Starting job: {job_name}\n")
        
        def run_job():
            try:
                import io
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                
                success = self.job_runner.run_job(job['config'], job_name)
                
                sys.stdout = old_stdout
                output = buffer.getvalue()
                
                self.root.after(0, lambda: self._job_complete(success, output))
            except Exception as e:
                self.root.after(0, lambda: self._job_complete(False, f"Error: {str(e)}"))
        
        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()
    
    def _job_complete(self, success: bool, output: str):
        """Handle job completion."""
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.progress_bar.stop()
        
        if success:
            self.progress_var.set("Completed successfully")
            self.status_text.insert(tk.END, "\n" + output)
            messagebox.showinfo("Success", "Job completed successfully!")
        else:
            self.progress_var.set("Failed")
            self.status_text.insert(tk.END, "\n" + output)
            messagebox.showerror("Error", "Job execution failed. Check the status log for details.")
    
    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo("About", "Fuzzy Matching Engine\n\nA configurable, high-performance fuzzy matching engine\nfor matching rows between CSV files or MySQL tables.")


def run_gui():
    """Launch the GUI application."""
    root = tk.Tk()
    app = MatchingEngineGUI(root)
    root.mainloop()


#!/usr/bin/env python3
"""
Optimized Single-Job Multi-Parameter URENIMOD Lookup Table Generator

This script:
1. Generates CSV file with parameter combinations
2. Submits ONE job that processes ALL parameter sets
3. Downloads all generated lookup tables

Key Advantage: Libraries are installed only once, then all parameter sets are processed
in the same container, making it much more time-efficient.

Author: Ryo Segawa
Date: June 2025
"""

import subprocess
import time
import json
import os
import sys
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

class GPULabSingleJobProcessor:
    def __init__(self):
        self.cert_path = "C:\\Users\\rsegawa\\login_ilabt_imec_be_rsegawa@ugent.be.pem"
        self.project = "urenimod"
        self.download_dir = "C:\\Users\\rsegawa\\OneDrive - UGent\\URENIMOD-data"
        self.csv_job_file = "job_parameters_generation.json"
        self.single_job_file = "job_single_multiparameter.json"
        # Ensure download directory exists
        Path(self.download_dir).mkdir(parents=True, exist_ok=True)
        
    def submit_job(self, job_content_str):
        """Submit a job to GPULab."""
        
        print("=== Submitting Job ===")
        print(f"Project: {self.project}")
        
        try:
            cmd = f'gpulab-cli --cert "{self.cert_path}" submit --project={self.project}'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                                  input=job_content_str, check=True)
            job_id = result.stdout.strip()
            print(f"✅ Job submitted successfully!")
            print(f"🆔 Job ID: {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"❌ Job submission failed: {e}")
            print(f"Error output: {e.stderr}")
            return None
        except Exception as e:
            print(f"❌ Error submitting job: {e}")
            return None
    
    def get_job_info(self, job_id):
        """Get detailed job information including status and container details."""
        
        list_cmd = f'gpulab-cli --cert "{self.cert_path}" jobs 2>NUL'
        
        try:
            list_result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, 
                                       encoding='utf-8', errors='replace')
            
            job_info = {}
            for line in list_result.stdout.split('\n'):
                if job_id in line or job_id[:8] in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        potential_status = parts[-1].strip()
                        if potential_status.upper() in ['RUNNING', 'FINISHED', 'FAILED', 'PENDING', 'WAITING', 'CANCELLED']:
                            job_info['status'] = potential_status.upper()
                        break
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LANG'] = 'en_US.UTF-8'
            
            cmd_list = ['gpulab-cli', '--cert', self.cert_path, 'jobs', job_id]
            result = subprocess.run(cmd_list, capture_output=True, text=True, 
                                   encoding='utf-8', errors='replace', timeout=60, env=env)
            
            all_output = result.stdout + "\n" + result.stderr
            
            for line in all_output.split('\n'):
                line = line.strip()
                if line.startswith('Status:'):
                    status = line.split(':', 1)[1].strip()
                    job_info['status'] = status.upper()
                elif line.startswith('SSH login::'):
                    ssh_info = line.split(':', 2)[2].strip()
                    ssh_match = re.search(r"ssh -i '[^']*' ([A-Za-z0-9]+)@([a-z0-9.]+)", ssh_info)
                    if ssh_match:
                        container_id = ssh_match.group(1)
                        ssh_host = ssh_match.group(2)
                        job_info['container'] = container_id
                        job_info['ssh_host'] = ssh_host
                        break
            
            if 'container' not in job_info and job_info.get('status') in ['RUNNING', 'FINISHED']:
                derived_container = job_id[:8]
                job_info['container'] = derived_container
            
            return job_info
            
        except Exception as e:
            print(f"❌ Exception in get_job_info: {e}")
            return {'status': 'ERROR'}
    
    def get_job_status(self, job_id):
        """Get the current status of the job."""
        job_info = self.get_job_info(job_id)
        return job_info.get('status', 'UNKNOWN')
    
    def get_job_logs(self, job_id):
        """Get the job logs to see progress."""
        
        cmd = f'gpulab-cli --cert "{self.cert_path}" log {job_id}'
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                                  check=True, encoding='utf-8', errors='replace')
            return result.stdout if result.stdout else ""
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Warning: Could not get logs - {e}")
            return ""
        except Exception as e:
            print(f"   ⚠️  Warning: Error reading logs - {e}")
            return ""
    
    def cancel_job(self, job_id):
        """Cancel the job to avoid waiting for the 30-minute timeout."""
        
        print(f"🛑 Cancelling job {job_id[:8]} to avoid timeout wait...")
        
        cmd = f'gpulab-cli --cert "{self.cert_path}" cancel {job_id}'
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                                  check=True, encoding='utf-8', errors='replace')
            print("✅ Job cancelled successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not cancel job - {e}")
            return False
        except Exception as e:
            print(f"⚠️  Warning: Error cancelling job - {e}")
            return False
    
    def check_computation_complete(self, job_id):
        """Check if the computation is complete by parsing logs for completion markers."""
        logs = self.get_job_logs(job_id)
        if not logs:
            return False
        
        completion_markers = [
            "All parameter sets processed successfully!",
            "Multi-parameter lookup generation completed!",
            "Starting 30-minute download window",
            "Starting sleep for download window",
            "sleep 1800",
            "Files are ready for download",
            "All lookup tables generated successfully"
        ]
        
        for marker in completion_markers:
            if marker in logs:
                print(f"🎯 Found completion marker: '{marker}'")
                return True
        
        if "All .pkl files ready" in logs or "parameter set" in logs and "completed" in logs:
            print(f"🎯 Found multi-parameter completion in logs")
            return True
        
        return False
    
    def wait_for_completion(self, job_id, timeout_minutes=60*24*7):
        """Wait for the job computation to complete."""
        
        print(f"=== Monitoring Job {job_id[:8]} ===")
        print(f"⏰ Timeout: {timeout_minutes} minutes")
        print("ℹ️  Note: Processing all parameter sets in single job")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        check_interval = 60*10  # Check every 10 minutes
        # check_interval = 30  # Check every 30 seconds
        
        print("⏳ Waiting for job to start...")
        
        while True:
            status = self.get_job_status(job_id)
            elapsed = int((time.time() - start_time) / 60)
            
            print(f"[{elapsed:02d}min] Job status: {status}")
            
            if status == "FINISHED":
                print("✅ Job completed successfully!")
                return True
            elif status == "RUNNING":
                print("🏃 Job is running...")
                if self.check_computation_complete(job_id):
                    print("✅ Computation completed! (Container still running for downloads)")
                    return True
                break
            elif status in ["FAILED", "CANCELLED", "ERROR"]:
                print(f"❌ Job failed with status: {status}")
                logs = self.get_job_logs(job_id)
                if logs and logs.strip():
                    print("📋 Last few lines of logs:")
                    recent_logs = logs.split('\n')[-5:]
                    for log_line in recent_logs:
                        if log_line.strip():
                            print(f"   {log_line}")
                return False
            elif status in ["UNKNOWN", "PENDING", "WAITING"]:
                print(f"⏳ Job status: {status} - waiting...")
                if status == "UNKNOWN" and elapsed > 3:
                    print("🔍 Status UNKNOWN but checking logs for completion...")
                    if self.check_computation_complete(job_id):
                        print("✅ Computation completed! (Status detection issue, but found in logs)")
                        return True
                elif status == "UNKNOWN" and elapsed >= 1:
                    print("🔍 Status UNKNOWN - checking logs for any activity...")
                    logs = self.get_job_logs(job_id)
                    if logs and logs.strip():
                        print("📋 Found job logs - job is likely running:")
                        recent_logs = logs.split('\n')[-5:]
                        for log_line in recent_logs:
                            if log_line.strip():
                                print(f"   {log_line}")
                        if self.check_computation_complete(job_id):
                            print("✅ Computation completed! (Found in logs)")
                            return True
            elif time.time() - start_time > timeout_seconds:
                print(f"⏰ Job timeout after {timeout_minutes} minutes")
                return False
            
            time.sleep(check_interval)
        
        print("📊 Monitoring job progress...")
        
        while True:
            status = self.get_job_status(job_id)
            elapsed = int((time.time() - start_time) / 60)
            
            print(f"[{elapsed:02d}min] Job status: {status}")
            
            if status == "FINISHED":
                print("✅ Job completed successfully!")
                return True
            elif status in ["FAILED", "CANCELLED", "ERROR"]:
                print(f"❌ Job failed with status: {status}")
                logs = self.get_job_logs(job_id)
                if logs and logs.strip():
                    print("📋 Last few lines of logs:")
                    recent_logs = logs.split('\n')[-5:]
                    for log_line in recent_logs:
                        if log_line.strip():
                            print(f"   {log_line}")
                return False
            elif status == "RUNNING":
                if self.check_computation_complete(job_id):
                    print("✅ Computation completed! (Container still running for downloads)")
                    return True
            elif time.time() - start_time > timeout_seconds:
                print(f"⏰ Job timeout after {timeout_minutes} minutes")
                return False
            
            # Show progress every 5 minutes for long job
            if elapsed % 5 == 0 and elapsed > 0:
                logs = self.get_job_logs(job_id)
                if logs and logs.strip():
                    recent_logs = logs.split('\n')[-10:]
                    print("📋 Recent activity:")
                    for log_line in recent_logs:
                        if log_line.strip():
                            print(f"   {log_line}")
                    if self.check_computation_complete(job_id):
                        print("✅ Computation completed! (Container still running for downloads)")
                        return True
                else:
                    print("📋 No recent logs available")
            
            time.sleep(check_interval)
    
    def download_lookup_files(self, job_id):
        """Download all generated lookup table files from the completed job."""
        
        print(f"📥 Downloading all lookup files from job {job_id[:8]}...")
        
        job_info = self.get_job_info(job_id)
        
        if 'container' not in job_info or 'ssh_host' not in job_info:
            print("❌ Could not extract container information from job")
            return False
        
        container = job_info['container']
        ssh_host = job_info['ssh_host']
        
        local_lookup_dir = Path(self.download_dir) / "lookup_tables" / "unmyelinated_axon"
        local_lookup_dir.mkdir(parents=True, exist_ok=True)
        
        # Search for all generated pickle files
        search_cmd = [
            "ssh", 
            "-i", self.cert_path,
            "-o", f"ProxyCommand=ssh -i {self.cert_path} fffrsegawau@bastion.ilabt.imec.be -W %h:%p",
            "-o", "StrictHostKeyChecking=no",
            f"{container}@{ssh_host}",
            "find /project_ghent/rsegawa/URENIMOD-lookups -name '*.pkl' 2>/dev/null"
        ]
        
        try:
            result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout.strip():
                pkl_files = result.stdout.strip().split('\n')
                print(f"🎯 Found {len(pkl_files)} pickle files to download")
                
                downloaded_count = 0
                for pkl_path in pkl_files:
                    if pkl_path.strip():
                        filename = pkl_path.split('/')[-1]
                        local_path = local_lookup_dir / filename
                        
                        print(f"📥 Downloading: {filename}")
                        
                        scp_cmd = [
                            "scp",
                            "-i", self.cert_path,
                            "-o", f"ProxyCommand=ssh -i {self.cert_path} fffrsegawau@bastion.ilabt.imec.be -W %h:%p",
                            "-o", "StrictHostKeyChecking=no",
                            f"{container}@{ssh_host}:{pkl_path}",
                            str(local_path)
                        ]
                        
                        try:
                            scp_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
                            
                            if scp_result.returncode == 0 and local_path.exists():
                                file_size = local_path.stat().st_size
                                print(f"✅ Downloaded: {filename} ({file_size} bytes)")
                                downloaded_count += 1
                            else:
                                print(f"❌ Failed to download {filename}")
                        except Exception as e:
                            print(f"❌ Error downloading {filename}: {e}")
                
                if downloaded_count > 0:
                    print(f"\n🎉 Successfully downloaded {downloaded_count} lookup table files!")
                    print(f"📁 Files saved to: {local_lookup_dir}")
                    return True
                else:
                    print("❌ No files were successfully downloaded")
                    return False
            else:
                print("❌ No pickle files found on server")
                return False
                
        except Exception as e:
            print(f"❌ Error searching for files: {e}")
            return False
    
    def get_single_job_config(self):
        """Read the single job configuration from JSON file."""
        with open(self.single_job_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_complete_workflow(self):
        """Run the complete workflow: generate CSV, then process all parameters in one job."""
        
        print("🚀 Starting URENIMOD Single-Job Multi-Parameter Generation")
        print("=" * 70)
        print("💡 This approach is much more efficient - libraries installed only once!")
        
        # Step 1: Submit CSV generation job
        print("\nStep 1: Generating parameters CSV file")
        print("-" * 40)
        
        with open(self.csv_job_file, 'r', encoding='utf-8') as f:
            csv_job_content = f.read()
        
        csv_job_id = self.submit_job(csv_job_content)
        if not csv_job_id:
            print("❌ Workflow failed at CSV job submission")
            return False

        # Wait for CSV file generation (simplified)
        print("⏳ Waiting for CSV file generation...")
        start_time = time.time()
        
        while time.time() - start_time < 60*360:  # 1 hour timeout
            status = self.get_job_status(csv_job_id)
            logs = self.get_job_logs(csv_job_id)
            
            if "CSV generation completed!" in logs and "lookup_parameters_" in logs:
                print("✅ CSV file generated successfully")
                break
            time.sleep(15)
        else:
            print("❌ CSV file generation timed out")
            return False

        # Get CSV file path on server
        today = datetime.now().strftime("%Y%m%d")
        csv_path = f"/project_ghent/rsegawa/URENIMOD-lookups/params_csv/lookup_parameters_{today}.csv"
        
        # Cancel CSV job
        print("Cancelling CSV generation job...")
        self.cancel_job(csv_job_id)

        # Step 2: Submit single multi-parameter job
        print(f"\nStep 2: Submitting single job to process all parameters")
        print("-" * 40)
        
        job_config = self.get_single_job_config()
        job_content = json.dumps(job_config, indent=2)
        
        job_id = self.submit_job(job_content)
        if not job_id:
            print("❌ Failed to submit multi-parameter job")
            return False

        # Step 3: Wait for completion
        print(f"\nStep 3: Waiting for multi-parameter job completion")
        print("-" * 40)
        
        completed = self.wait_for_completion(job_id, timeout_minutes=60*24*7)  # 7 days timeout
        if not completed:
            print("❌ Multi-parameter job failed or timed out")
            return False

        # Step 4: Download all files
        print(f"\nStep 4: Downloading all generated lookup tables")
        print("-" * 40)
        
        downloaded = self.download_lookup_files(job_id)
        
        if downloaded:
            print("\n🎉 SINGLE-JOB WORKFLOW COMPLETED SUCCESSFULLY!")
            print("✅ All parameter sets processed in one efficient job")
            print(f"📁 Lookup tables downloaded to: {self.download_dir}")
            
            # Cancel the job to free resources
            self.cancel_job(job_id)
            
            print("\n💡 Much faster than multiple jobs - libraries installed only once!")
            return True
        else:
            print("\n⚠️  Job completed but download failed")
            return False

def main():
    """Main function to run the single-job processor."""
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("=== URENIMOD Single-Job Multi-Parameter Generator ===")
    print(f"Working directory: {script_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if required job files exist
    job_files = ["job_parameters_generation.json"]
    missing_files = [f for f in job_files if not Path(f).exists()]
    if missing_files:
        print("❌ Required job files not found:")
        for f in missing_files:
            print(f"   - {f}")
        print("Please run this script from the GPULab directory.")
        return
    
    # Initialize and run
    processor = GPULabSingleJobProcessor()
    success = processor.run_complete_workflow()
    
    if success:
        print("\n🎊 All done! Much more efficient than multiple jobs!")
    else:
        print("\n💔 Workflow had issues, but don't worry - we can troubleshoot!")

if __name__ == "__main__":
    main()
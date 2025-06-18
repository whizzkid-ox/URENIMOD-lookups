#!/usr/bin/env python3
"""
Modified Auto Submit and Download Script for URENIMOD Lookup Tables

This script:
1. Generates CSV file with parameter combinations
2. Reads the CSV and submits lookup table jobs for each parameter set
3. Monitors job progress and downloads results

Key Changes:
- CSV generation followed by individual lookup jobs for each parameter set
- Variable substitution in command instead of hardcoded values
- Batch processing of multiple parameter combinations

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

class GPULabMultiParameterProcessor:
    def __init__(self):
        self.cert_path = "C:\\Users\\rsegawa\\login_ilabt_imec_be_rsegawa@ugent.be.pem"
        self.project = "urenimod"
        self.download_dir = "C:\\Users\\rsegawa\\OneDrive - UGent\\URENIMOD-data"
        self.csv_job_file = "job_parameters_generation.json"
        self.lookup_job_template_file = "job_LUT_auto_download.json"
        self.csv_file_path = None
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
        
        # Try the jobs command without job ID first to get status from list
        list_cmd = f'gpulab-cli --cert "{self.cert_path}" jobs 2>NUL'
        
        try:
            list_result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, 
                                       encoding='utf-8', errors='replace')
            
            job_info = {}
            # Parse the jobs list to find our job and its status
            for line in list_result.stdout.split('\n'):
                # Match specifically by our job ID (full or first 8 characters)
                if job_id in line or job_id[:8] in line:
                    print(f"🎯 Found job line: {line}")
                    # The line format appears to be columnar with STATUS at the end
                    # Split by whitespace and take the last non-empty part as status
                    parts = line.split()
                    if len(parts) >= 2:
                        # Last part should be the status
                        potential_status = parts[-1].strip()
                        if potential_status.upper() in ['RUNNING', 'FINISHED', 'FAILED', 'PENDING', 'WAITING', 'CANCELLED']:
                            job_info['status'] = potential_status.upper()
                            print(f"🎯 Found status from list: '{potential_status.upper()}'")
                        break
            
            # Get detailed job information for container and SSH info
            print("🔍 Getting detailed job information...")
            
            # Method with environment fix for encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LANG'] = 'en_US.UTF-8'
            
            cmd_list = ['gpulab-cli', '--cert', self.cert_path, 'jobs', job_id]
            result = subprocess.run(cmd_list, capture_output=True, text=True, 
                                   encoding='utf-8', errors='replace', timeout=60, env=env)
            
            # Parse all available output for SSH info and status
            all_output = result.stdout + "\n" + result.stderr
            
            for line in all_output.split('\n'):
                line = line.strip()
                if line.startswith('Status:'):
                    status = line.split(':', 1)[1].strip()
                    job_info['status'] = status.upper()
                    print(f"🎯 Found status from job details: '{status.upper()}'")
                elif line.startswith('SSH login::'):
                    # Extract container ID and SSH host from SSH login line
                    ssh_info = line.split(':', 2)[2].strip()
                    
                    ssh_match = re.search(r"ssh -i '[^']*' ([A-Za-z0-9]+)@([a-z0-9.]+)", ssh_info)
                    if ssh_match:
                        container_id = ssh_match.group(1)
                        ssh_host = ssh_match.group(2)
                        job_info['container'] = container_id
                        job_info['ssh_host'] = ssh_host
                        print(f"🎯 Extracted container ID: '{container_id}'")
                        print(f"🎯 Extracted SSH host: '{ssh_host}'")
                        break
            
            # If we still don't have container info, try to derive it from job_id
            if 'container' not in job_info and job_info.get('status') in ['RUNNING', 'FINISHED']:
                print("🔍 No container found in output, trying to derive from job ID...")
                derived_container = job_id[:8]
                job_info['container'] = derived_container
                print(f"🎯 Using derived container ID: '{derived_container}'")
            
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
        
        # Look for completion indicators in the logs
        completion_markers = [
            "Lookup generation completed successfully!",
            "All lookup tables generated successfully!",
            "Lookup table generation complete",
            "Starting 30-minute download window",
            "Starting sleep for download window",
            "sleep 1800",  # The 30-minute sleep command
            "Files are ready for download",
            "Generated lookup tables successfully"
        ]
        
        for marker in completion_markers:
            if marker in logs:
                print(f"🎯 Found completion marker: '{marker}'")
                return True
        
        # Also check for the specific pattern we've seen in logs
        if "All .pkl files ready" in logs or "Saving to" in logs and ".pkl" in logs:
            print(f"🎯 Found pickle file generation completion in logs")
            return True
        
        return False
    
    def wait_for_completion(self, job_id, timeout_minutes=60):
        """Wait for the job computation to complete (not the container to finish)."""
        
        print(f"=== Monitoring Job {job_id[:8]} ===")
        print(f"⏰ Timeout: {timeout_minutes} minutes")
        print("ℹ️  Note: Looking for computation completion, not container finish")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        check_interval = 30  # Check every 30 seconds
        
        # First wait for job to start running
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
                # Check if computation is actually complete even though status is RUNNING
                if self.check_computation_complete(job_id):
                    print("✅ Computation completed! (Container still running for downloads)")
                    return True
                break  # Job is running, continue monitoring
            elif status in ["FAILED", "CANCELLED", "ERROR"]:
                print(f"❌ Job failed with status: {status}")
                # Show logs to help debug
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
                # If status is UNKNOWN, also check if computation might be complete
                if status == "UNKNOWN" and elapsed > 3:  # After 3+ minutes, check logs
                    print("🔍 Status UNKNOWN but checking logs for completion...")
                    if self.check_computation_complete(job_id):
                        print("✅ Computation completed! (Status detection issue, but found in logs)")
                        return True
                elif status == "UNKNOWN" and elapsed >= 1:  # Check logs early for UNKNOWN status
                    print("🔍 Status UNKNOWN - checking logs for any activity...")
                    logs = self.get_job_logs(job_id)
                    if logs and logs.strip():
                        print("📋 Found job logs - job is likely running:")
                        recent_logs = logs.split('\n')[-5:]
                        for log_line in recent_logs:
                            if log_line.strip():
                                print(f"   {log_line}")
                        # Check for completion
                        if self.check_computation_complete(job_id):
                            print("✅ Computation completed! (Found in logs)")
                            return True
            elif time.time() - start_time > timeout_seconds:
                print(f"⏰ Job timeout after {timeout_minutes} minutes")
                return False
            
            time.sleep(check_interval)
        
        # Now monitor until computation completion
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
                # Show logs to help debug
                logs = self.get_job_logs(job_id)
                if logs and logs.strip():
                    print("📋 Last few lines of logs:")
                    recent_logs = logs.split('\n')[-5:]
                    for log_line in recent_logs:
                        if log_line.strip():
                            print(f"   {log_line}")
                return False
            elif status == "RUNNING":
                # Check if computation is complete by analyzing logs
                if self.check_computation_complete(job_id):
                    print("✅ Computation completed! (Container still running for downloads)")
                    return True
            elif time.time() - start_time > timeout_seconds:
                print(f"⏰ Job timeout after {timeout_minutes} minutes")
                return False
            
            # Show progress every 2 minutes and check for completion
            if elapsed % 2 == 0 and elapsed > 0:
                logs = self.get_job_logs(job_id)
                if logs and logs.strip():
                    recent_logs = logs.split('\n')[-10:]  # Last 10 lines
                    print("📋 Recent activity:")
                    for log_line in recent_logs:
                        if log_line.strip():
                            print(f"   {log_line}")
                    # Check for completion in the recent logs
                    if self.check_computation_complete(job_id):
                        print("✅ Computation completed! (Container still running for downloads)")
                        return True
                else:
                    print("📋 No recent logs available")
            
            time.sleep(check_interval)
    
    def download_lookup_files(self, job_id):
        """Download the generated lookup table files from a completed job."""
        
        print(f"📥 Downloading lookup files from job {job_id[:8]}...")
        
        # Get job info to extract container details
        job_info = self.get_job_info(job_id)
        
        if 'container' not in job_info or 'ssh_host' not in job_info:
            print("❌ Could not extract container information from job")
            return False
        
        container = job_info['container']
        ssh_host = job_info['ssh_host']
        
        print(f"📡 Container: {container}")
        print(f"🖥️  SSH Host: {ssh_host}")
        
        # Set up local download directory
        local_lookup_dir = Path(self.download_dir) / "lookup_tables" / "unmyelinated_axon"
        local_lookup_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate the date-based remote directory path
        today = datetime.now()
        date_folder = f"{today.year:04d}{today.month:02d}{today.day:02d}"
        remote_dir = f"/project_ghent/rsegawa/URENIMOD-lookups/NME/lookup_tables/unmyelinated_fiber_{date_folder}"
        
        print(f"🔍 Checking for files in {remote_dir}...")
        
        # Check for pickle files in the remote directory
        ssh_check_cmd = [
            "ssh",
            "-i", self.cert_path,
            "-o", f"ProxyCommand=ssh -i {self.cert_path} fffrsegawau@bastion.ilabt.imec.be -W %h:%p",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15",
            f"{container}@{ssh_host}",
            f"ls -la {remote_dir}/*.pkl 2>/dev/null || echo 'No pkl files found'"
        ]
        
        try:
            result = subprocess.run(ssh_check_cmd, capture_output=True, text=True, timeout=45)
            
            print(f"🔍 SSH command output:")
            print(f"Return code: {result.returncode}")
            
            # Check for SSH connection errors first
            if result.returncode != 0:
                if "Permission denied" in result.stderr or "Permission denied" in result.stdout:
                    print("🔐 SSH Permission Error - cannot connect to remote server")
                    return False
                elif "Connection" in result.stderr or "timeout" in result.stderr.lower():
                    print("🌐 SSH Connection Error - cannot reach remote server")
                    return False
                elif "No such file or directory" in result.stdout or "No pkl files found" in result.stdout:
                    print("📁 No pickle files found in expected directory")
                    # Try to find files in alternative locations
                    print("🔍 Searching for pickle files in alternative locations...")
                    search_cmd = ssh_check_cmd[:-1] + ["find /project_ghent/rsegawa/URENIMOD-lookups -name '*.pkl' 2>/dev/null | head -5"]
                    search_result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=30)
                    if search_result.returncode == 0 and search_result.stdout.strip():
                        print(f"🎯 Found pickle files elsewhere:")
                        print(search_result.stdout)
                        # Use the first file found for download
                        first_file = search_result.stdout.strip().split('\n')[0]
                        pkl_files = [(first_file, first_file.split('/')[-1])]
                        print(f"🎯 Will download: {first_file}")
                    else:
                        print("❌ No pickle files found in the entire project directory")
                        return False
                else:
                    print(f"❌ SSH command failed with return code {result.returncode}")
                    if result.stderr:
                        print(f"   Error details: {result.stderr}")
                    return False
            else:
                # Process successful output
                stdout_clean = result.stdout.strip()
                
                # Handle different cases of "no files found"
                no_files_indicators = [
                    "No pkl files found",
                    "No such file or directory",
                    "cannot access",
                ]
                has_pkl_files = (stdout_clean and
                               not any(indicator in result.stdout for indicator in no_files_indicators) and
                               ".pkl" in result.stdout)
                
                if not has_pkl_files:
                    print("❌ No pickle files detected in expected directory")
                    # Try alternative search
                    print("🔍 Searching for pickle files in alternative locations...")
                    search_cmd = ssh_check_cmd[:-1] + ["find /project_ghent/rsegawa/URENIMOD-lookups -name '*.pkl' 2>/dev/null | head -5"]
                    search_result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=30)
                    if search_result.returncode == 0 and search_result.stdout.strip():
                        print(f"🎯 Found pickle files elsewhere:")
                        print(search_result.stdout)
                        # Store full paths for alternative files
                        found_files = search_result.stdout.strip().split('\n')
                        pkl_files = []
                        for full_path in found_files:
                            if full_path.strip():
                                filename = full_path.split('/')[-1]
                                pkl_files.append((full_path, filename))
                                print(f"📥 Will download: {filename} from {full_path}")
                    else:
                        print("❌ No pickle files found anywhere")
                        return False
                else:
                    print("📁 Available files:")
                    print(result.stdout)
                    # Extract filenames from ls output
                    pkl_files = []
                    for line in result.stdout.split('\n'):
                        if '.pkl' in line and not line.startswith('total'):
                            # Extract filename from ls -la output
                            parts = line.split()
                            if len(parts) >= 9:
                                filename = ' '.join(parts[8:])  # Handle filenames with spaces
                                if '/' in filename:
                                    filename = filename.split('/')[-1]  # Get just the filename part
                                # Store as tuple: (None, filename) - None means use remote_dir
                                pkl_files.append((None, filename))
                    
                    if not pkl_files:
                        print("❌ No pickle files detected in listing")
                        return False
            
            print(f"🎯 Found {len(pkl_files)} pickle file(s) to download")
            
        except subprocess.TimeoutExpired:
            print("⏰ SSH connection timed out")
            return False
        except Exception as e:
            print(f"❌ SSH check failed: {e}")
            return False
        
        # Download each pickle file using SCP
        downloaded_count = 0
        for pkl_item in pkl_files:
            # Handle both tuple format (full_path, filename) and string format
            if isinstance(pkl_item, tuple):
                full_path, filename = pkl_item
                if full_path:  # Alternative location - use full path
                    remote_path = full_path
                else:  # Regular location - use remote_dir + filename
                    remote_path = f"{remote_dir}/{filename}"
                local_filename = filename
            else:
                # Legacy string format
                remote_path = f"{remote_dir}/{pkl_item}"
                local_filename = pkl_item
            
            print(f"\n📥 Downloading: {local_filename}")
            print(f"   Remote path: {remote_path}")
            
            local_path = local_lookup_dir / local_filename
            
            scp_cmd = [
                "scp",
                "-i", self.cert_path,
                "-o", f"ProxyCommand=ssh -i {self.cert_path} fffrsegawau@bastion.ilabt.imec.be -W %h:%p",
                "-o", "StrictHostKeyChecking=no",
                f"{container}@{ssh_host}:{remote_path}",
                str(local_path)
            ]
            
            try:
                result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0 and local_path.exists():
                    file_size = local_path.stat().st_size
                    print(f"✅ Downloaded: {local_filename} ({file_size} bytes)")
                    
                    # Verify the pickle file
                    try:
                        import pickle
                        with open(local_path, 'rb') as f:
                            data = pickle.load(f)
                        print(f"   ✅ Pickle file verified")
                        downloaded_count += 1
                    except Exception as e:
                        print(f"   ⚠️  Warning: Could not verify pickle file: {e}")
                        downloaded_count += 1  # Still count it as downloaded
                
                else:
                    print(f"❌ Failed to download {local_filename}")
                    if result.stderr:
                        print(f"   Error: {result.stderr}")
                        
            except subprocess.TimeoutExpired:
                print(f"⏰ Download of {local_filename} timed out")
            except Exception as e:
                print(f"❌ Error downloading {local_filename}: {e}")
        
        if downloaded_count > 0:
            print(f"\n🎉 Successfully downloaded {downloaded_count} file(s) to:")
            print(f"📁 {local_lookup_dir}")
            return True
        else:
            print("❌ No files were successfully downloaded")
            return False
        
        # Now monitor until computation completion
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
                return False
            elif status == "RUNNING":
                # Check if computation is complete by analyzing logs
                if self.check_computation_complete(job_id):
                    print("✅ Computation completed!")
                    return True
            elif time.time() - start_time > timeout_seconds:
                print(f"⏰ Job timeout after {timeout_minutes} minutes")
                return False
            
            time.sleep(check_interval)
    
    def check_csv_completion(self, job_id):
        """Check if CSV file has been generated."""
        logs = self.get_job_logs(job_id)
        if not logs:
            return False, None
        
        print(f"🔍 Checking logs for CSV completion...")
        
        # Look for completion indicators in the logs
        csv_markers = [
            "CSV generation completed!",
            "CSV test completed!",
            "lookup_parameters_",
            ".csv"
        ]
        
        # Check if we have the basic completion markers
        has_completion = any(marker in logs for marker in ["CSV generation completed!", "CSV test completed!"])
        has_csv_file = "lookup_parameters_" in logs and ".csv" in logs
        
        print(f"   Has completion marker: {has_completion}")
        print(f"   Has CSV file reference: {has_csv_file}")
        
        if has_completion and has_csv_file:
            print("✅ CSV generation detected as complete")
            # Extract CSV file path from logs
            for line in logs.split('\n'):
                if 'lookup_parameters_' in line and '.csv' in line:
                    # Try to extract the full path
                    if '/project_ghent/rsegawa/URENIMOD-lookups/params_csv_test/' in line:
                        csv_path = line.strip()
                        print(f"   Found CSV path in logs: {csv_path}")
                        return True, csv_path
            
            # If we found completion but no specific path, still return True
            print("   CSV completed but no specific path found in logs")
            return True, None
        
        # Also check for the specific file listing command output
        if "Generated CSV file:" in logs or "/project_ghent/rsegawa/URENIMOD-lookups/params_csv_test/lookup_parameters_" in logs:
            print("✅ CSV file found in logs")
            return True, None
        
        return False, None
    
    def get_csv_file_path(self, job_id):
        """Get the path to the generated CSV file on the server."""
        
        print(f"🔍 Looking for CSV file from job {job_id[:8]}...")
        
        # The CSV file should be at a predictable location
        today = datetime.now().strftime("%Y%m%d")
        expected_csv_path = f"/project_ghent/rsegawa/URENIMOD-lookups/params_csv_test/lookup_parameters_{today}.csv"
        
        print(f"📁 Expected CSV path: {expected_csv_path}")
        return expected_csv_path
    
    def generate_lookup_job_config(self, params, csv_file_path):
        """Generate job configuration for a specific parameter set."""
        
        # Read the template job file
        with open(self.lookup_job_template_file, 'r') as f:
            job_template = json.load(f)
        
        # Extract parameters
        fiber_length = params['fiber_length']
        fiber_diameter = params['fiber_diameter']
        membrane_thickness = params['membrane_thickness']
        freq = params['freq']
        amp = params['amp']
        charge = params['charge']
        
        # Create parameter-specific job name
        job_name = f"lookup_fl{fiber_length:.1e}_fd{fiber_diameter:.1e}_freq{freq:.1e}_amp{amp:.1e}_ch{charge:.1f}"
        job_template['name'] = job_name
        
        # Modify the command to use variables instead of hardcoded values
        original_command = job_template['request']['docker']['command']
        
        # Replace the hardcoded python run_lookups.py command with parameterized version
        new_command = original_command.replace(
            'python run_lookups.py -fiber_length 1e-3 -fiber_diameter 1e-6 -membrane_thickness 1.4e-9 -freq 1e6 -amp 1e6 -charge -65.0',
            f'python run_lookups.py -fiber_length {fiber_length} -fiber_diameter {fiber_diameter} -membrane_thickness {membrane_thickness} -freq {freq} -amp {amp} -charge {charge}'
        )
        
        job_template['request']['docker']['command'] = new_command
        
        return job_template
    
    def read_csv_from_server_via_ssh(self, csv_job_id, csv_file_path):
        """Read CSV file directly from server via SSH and return the parameter data."""
        
        print(f"📊 Reading CSV file from server: {csv_file_path}")
        
        # Get job info to extract container details from the CSV job
        job_info = self.get_job_info(csv_job_id)
        
        if 'container' not in job_info:
            print("❌ Could not extract container information from CSV job")
            return None
        
        container = job_info['container']
        ssh_host = job_info.get('ssh_host', 'default.gpulab.ilabt.imec.be')
        
        # Read CSV file via SSH
        ssh_cmd = [
            "ssh",
            "-i", self.cert_path,
            "-o", f"ProxyCommand=ssh -i {self.cert_path} fffrsegawau@bastion.ilabt.imec.be -W %h:%p",
            "-o", "StrictHostKeyChecking=no",
            f"{container}@{ssh_host}",
            f"cat {csv_file_path}"
        ]
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout.strip():
                csv_content = result.stdout.strip()
                print(f"✅ Successfully read CSV file ({len(csv_content)} characters)")
                
                # Parse CSV content
                import io
                import pandas as pd
                
                df = pd.read_csv(io.StringIO(csv_content))
                print(f"📊 Loaded {len(df)} parameter combinations from server")
                return df
            else:
                print(f"❌ Failed to read CSV file from server")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ SSH read timed out")
            return None
        except Exception as e:
            print(f"❌ Error reading CSV file via SSH: {e}")
            return None
    
    def run_complete_workflow(self):
        """Run the complete workflow: generate CSV, then process each parameter set."""
        
        print("🚀 Starting URENIMOD Multi-Parameter Lookup Table Generation")
        print("=" * 70)
        
        # Step 1: Submit CSV generation job
        print("\nStep 1: Generating parameters CSV file")
        print("-" * 40)
        
        with open(self.csv_job_file, 'r') as f:
            csv_job_content = f.read()
        
        csv_job_id = self.submit_job(csv_job_content)
        if not csv_job_id:
            print("❌ Workflow failed at CSV job submission")
            return False

        # Wait for CSV file generation
        print("⏳ Waiting for CSV file generation...")
        start_time = time.time()
        csv_path = None
        check_count = 0
        
        while time.time() - start_time < 600:  # 10 minutes timeout
            check_count += 1
            elapsed = int((time.time() - start_time) / 60)
            
            print(f"\n[{elapsed:02d}min] Check #{check_count}: Looking for CSV completion...")
            
            # Check job status first
            status = self.get_job_status(csv_job_id)
            print(f"   Job Status: {status}")
            
            completed, csv_remote_path = self.check_csv_completion(csv_job_id)
            if completed:
                print("✅ CSV file generation detected as complete!")
                # Get the CSV file path on server
                csv_path = self.get_csv_file_path(csv_job_id)
                break
            else:
                print(f"   CSV not yet complete, waiting 15 seconds...")
                # Show recent logs to debug
                logs = self.get_job_logs(csv_job_id)
                if logs:
                    recent_logs = logs.split('\n')[-3:]  # Last 3 lines
                    print(f"   Recent logs:")
                    for log_line in recent_logs:
                        if log_line.strip():
                            print(f"     {log_line}")
                else:
                    print(f"     No logs available yet")
            
            time.sleep(15)
        else:
            elapsed = int((time.time() - start_time) / 60)
            print(f"\n❌ CSV file generation timed out after {elapsed} minutes")
            # Show final logs for debugging
            logs = self.get_job_logs(csv_job_id)
            if logs:
                print("Final logs:")
                final_logs = logs.split('\n')[-10:]  # Last 10 lines
                for log_line in final_logs:
                    if log_line.strip():
                        print(f"   {log_line}")
            return False

        if not csv_path:
            print("❌ Failed to get CSV file path")
            return False

        # Step 2: Read CSV file from server and process each parameter set
        print(f"\nStep 2: Reading parameter combinations from server")
        print(f"CSV path on server: {csv_path}")
        print("-" * 40)
        
        # Read CSV directly from server via SSH
        df = self.read_csv_from_server_via_ssh(csv_job_id, csv_path)
        if df is None:
            print("❌ Failed to read CSV file from server")
            return False
        
        # Show sample of parameters
        print("\nSample parameters:")
        print(df.head())

        # Cancel CSV job since we don't need it anymore
        print("\nCancelling CSV generation job...")
        self.cancel_job(csv_job_id)

        # Step 3: Process each parameter set
        print(f"\nStep 3: Processing {len(df)} parameter combinations")
        print("-" * 40)
        
        successful_jobs = 0
        failed_jobs = 0
        
        for idx, params in df.iterrows():
            print(f"\n🔄 Processing parameter set {idx + 1}/{len(df)}")
            print(f"   fiber_length={params['fiber_length']:.1e}, fiber_diameter={params['fiber_diameter']:.1e}")
            print(f"   membrane_thickness={params['membrane_thickness']:.1e}, freq={params['freq']:.1e}")
            print(f"   amp={params['amp']:.1e}, charge={params['charge']:.1f}")
            
            # Generate job configuration for this parameter set
            job_config = self.generate_lookup_job_config(params, csv_path)
            job_content = json.dumps(job_config, indent=2)
            
            # Submit the job
            job_id = self.submit_job(job_content)
            if not job_id:
                print(f"❌ Failed to submit job for parameter set {idx + 1}")
                failed_jobs += 1
                continue
            
            # Wait for completion with shorter timeout for individual jobs
            print(f"⏳ Waiting for job {job_id[:8]} to complete...")
            completed = self.wait_for_completion(job_id, timeout_minutes=20)
            
            if completed:
                print(f"✅ Parameter set {idx + 1} completed successfully")
                
                # Download the generated lookup files
                print(f"📥 Downloading lookup files for parameter set {idx + 1}...")
                download_success = self.download_lookup_files(job_id)
                
                if download_success:
                    print(f"✅ Lookup files downloaded successfully for parameter set {idx + 1}")
                    successful_jobs += 1
                else:
                    print(f"⚠️  Parameter set {idx + 1} completed but download failed")
                    # Still count as successful since computation completed
                    successful_jobs += 1
                
                # Cancel job to free resources
                self.cancel_job(job_id)
            else:
                print(f"❌ Parameter set {idx + 1} failed or timed out")
                failed_jobs += 1
                # Try to cancel failed job
                self.cancel_job(job_id)
            
            # Small delay between jobs to avoid overwhelming the system
            if idx < len(df) - 1:  # Don't sleep after the last job
                print("⏸️  Waiting 30 seconds before next job...")
                time.sleep(30)

        # Step 4: Summary
        print(f"\n🎉 MULTI-PARAMETER WORKFLOW COMPLETED!")
        print("=" * 50)
        print(f"✅ Successful jobs: {successful_jobs}")
        print(f"❌ Failed jobs: {failed_jobs}")
        print(f"📊 Total processed: {successful_jobs + failed_jobs}/{len(df)}")
        
        if successful_jobs > 0:
            print(f"📁 Lookup tables generated and downloaded to:")
            print(f"   {self.download_dir}\\lookup_tables\\unmyelinated_axon\\")
            print(f"💡 You can now use these lookup tables in your URENIMOD research!")
            
            # List downloaded files
            lookup_dir = Path(self.download_dir) / "lookup_tables" / "unmyelinated_axon"
            if lookup_dir.exists():
                pkl_files = list(lookup_dir.glob("*.pkl"))
                if pkl_files:
                    print(f"\n📋 Downloaded lookup table files ({len(pkl_files)} files):")
                    for pkl_file in pkl_files:
                        size = pkl_file.stat().st_size
                        mod_time = datetime.fromtimestamp(pkl_file.stat().st_mtime)
                        print(f"   📦 {pkl_file.name} ({size} bytes, {mod_time.strftime('%H:%M')})")
            
            return True
        else:
            print("⚠️  No jobs completed successfully")
            return False

def main():
    """Main function to run the multi-parameter processor."""
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("=== URENIMOD Multi-Parameter Lookup Table Generator ===")
    print(f"Working directory: {script_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if required job files exist
    job_files = ["job_parameters_generation.json", "job_LUT_auto_download.json"]
    missing_files = [f for f in job_files if not Path(f).exists()]
    if missing_files:
        print("❌ Required job files not found:")
        for f in missing_files:
            print(f"   - {f}")
        print("Please run this script from the GPULab directory.")
        return
    
    # Initialize and run
    processor = GPULabMultiParameterProcessor()
    success = processor.run_complete_workflow()
    
    if success:
        print("\n🎊 All done! Happy researching with URENIMOD!")
    else:
        print("\n💔 Workflow had issues, but don't worry - we can troubleshoot!")

if __name__ == "__main__":
    main()
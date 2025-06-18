#!/usr/bin/env python3
"""
Auto Submit and Download Script for URENIMOD Lookup Tables

This script:
1. Submits the lookup table generation job to GPULab
2. Monitors job progress (detects computation completion, not just container finish)
3. Automatically downloads the generated pickle files to OneDrive
4. Uses the SSH connection method we established

Key Features:
- Handles Unicode errors in subprocess output (from emojis in gpulab-cli)
- Detects computation completion by parsing logs (not waiting for container to finish)
- GPULab containers stay "RUNNING" for 30 minutes after computation for file downloads

Author: Ryo Segawa
Date: June 2025
"""

import subprocess
import time
import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime

class GPULabAutoDownloader:
    def __init__(self):
        self.cert_path = "C:\\Users\\rsegawa\\login_ilabt_imec_be_rsegawa@ugent.be.pem"
        self.project = "urenimod"
        self.download_dir = "C:\\Users\\rsegawa\\OneDrive - UGent\\URENIMOD-data"
        self.job_file = "gpulab_lookup_with_auto_download.json"        # Ensure download directory exists
        Path(self.download_dir).mkdir(parents=True, exist_ok=True)
        
    def submit_job(self):
        """Submit the lookup generation job to GPULab."""
        
        print("=== Submitting URENIMOD Lookup Table Job ===")
        print(f"Job file: {self.job_file}")
        print(f"Project: {self.project}")
        
        # Read the job file content and pipe it to gpulab-cli
        try:
            with open(self.job_file, 'r') as f:
                job_content = f.read()
            
            cmd = f'gpulab-cli --cert "{self.cert_path}" submit --project={self.project}'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                                  input=job_content, check=True)
            job_id = result.stdout.strip()
            print(f"✅ Job submitted successfully!")
            print(f"🆔 Job ID: {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"❌ Job submission failed: {e}")
            print(f"Error output: {e.stderr}")
            return None
        except Exception as e:
            print(f"❌ Error reading job file: {e}")
            return None
    
    def get_job_info(self, job_id):
        """Get detailed job information including status and container details."""
        
        # Try the jobs command without job ID first to get status from list
        list_cmd = f'gpulab-cli --cert "{self.cert_path}" jobs 2>NUL'
        
        try:
            list_result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, 
                                       encoding='utf-8', errors='replace')
            
            print(f"🔍 DEBUG: Jobs list output:")
            print(f"Return code: {list_result.returncode}")
            print(f"Output length: {len(list_result.stdout)} chars")
            if list_result.stdout.strip():
                print("--- START JOBS LIST ---")
                print(list_result.stdout)
                print("--- END JOBS LIST ---")
              
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
                              # Extract CID (Container ID) from the jobs list line
                            # CID is typically in a dedicated column before STATUS
                            try:
                                # Find the CID column (usually 2nd to last or 3rd to last)
                                for i, part in enumerate(parts):
                                    if part.isdigit() and len(part) <= 3:  # CID is usually a short number
                                        # Check if next part is the status
                                        if i < len(parts) - 1 and parts[i + 1].upper() in ['RUNNING', 'FINISHED', 'FAILED', 'PENDING', 'WAITING', 'CANCELLED']:
                                            # Note: This is cluster ID, not container ID - detailed job query will get the real container ID
                                            job_info['cluster_id'] = part  
                                            print(f"🎯 Extracted cluster ID from jobs list: '{part}'")
                                            break
                            except Exception as e:
                                print(f"⚠️ Could not extract cluster ID from line: {e}")
                            break            # Always try to get detailed job information for container and SSH info
            print("🔍 Getting detailed job information...")
            
            # Method 1: Using cmd as list with environment fix for encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LANG'] = 'en_US.UTF-8'
            
            cmd_list = ['gpulab-cli', '--cert', self.cert_path, 'jobs', job_id]
            result1 = subprocess.run(cmd_list, capture_output=True, text=True, 
                                   encoding='utf-8', errors='replace', timeout=60, env=env)
            
            print(f"🔍 Method 1 (cmd list with env) - Return code: {result1.returncode}, Output: {len(result1.stdout)} chars")
            
            # Method 2: Using shell=True with explicit command and chcp to change codepage
            cmd_str = f'chcp 65001 >nul 2>&1 && gpulab-cli --cert "{self.cert_path}" jobs {job_id}'
            result2 = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, 
                                   encoding='utf-8', errors='replace', timeout=60)
            
            print(f"🔍 Method 2 (shell with chcp) - Return code: {result2.returncode}, Output: {len(result2.stdout)} chars")
            
            # Method 3: Try redirecting stderr to capture all output
            cmd_str3 = f'gpulab-cli --cert "{self.cert_path}" jobs {job_id} 2>&1'
            result3 = subprocess.run(cmd_str3, shell=True, capture_output=True, text=True, 
                                   encoding='utf-8', errors='replace', timeout=60)
            
            print(f"🔍 Method 3 (redirect stderr) - Return code: {result3.returncode}, Output: {len(result3.stdout)} chars")
            
            # Use the result with the most output
            results = [(result1, "Method 1"), (result2, "Method 2"), (result3, "Method 3")]
            result, method_name = max(results, key=lambda x: len(x[0].stdout))
            print(f"🎯 Using {method_name} result")
            
            print(f"🔍 DEBUG: Individual job output:")
            print(f"Return code: {result.returncode}")
            print(f"Output length: {len(result.stdout)} chars")
            print(f"Error length: {len(result.stderr)} chars")
            if result.stdout.strip():
                print("--- START INDIVIDUAL JOB ---")
                print(result.stdout)
                print("--- END INDIVIDUAL JOB ---")
            if result.stderr.strip():
                print("--- STDERR ---")
                print(result.stderr)
                print("--- END STDERR ---")
            
            # Always parse all available output for SSH info and status
            all_output = result.stdout + "\n" + result.stderr
            parsed_ssh_info = False
            
            for line in all_output.split('\n'):
                line = line.strip()
                if line.startswith('Status:'):
                    status = line.split(':', 1)[1].strip()
                    job_info['status'] = status.upper()
                    print(f"🎯 Found status from job details: '{status.upper()}'")
                elif line.startswith('SSH login::'):
                    # Extract container ID and SSH host from SSH login line
                    # Format: ssh -i 'path' CONTAINER@HOST -oProxyCommand=...
                    ssh_info = line.split(':', 2)[2].strip()  # Get everything after "SSH login::"
                    
                    import re
                    ssh_match = re.search(r"ssh -i '[^']*' ([A-Za-z0-9]+)@([a-z0-9.]+)", ssh_info)
                    if ssh_match:
                        container_id = ssh_match.group(1)
                        ssh_host = ssh_match.group(2)
                        job_info['container'] = container_id
                        job_info['ssh_host'] = ssh_host
                        print(f"🎯 Extracted container ID from SSH login: '{container_id}'")
                        print(f"🎯 Extracted SSH host: '{ssh_host}'")
                        parsed_ssh_info = True
                        break
                    else:
                        print(f"⚠️ Could not parse SSH login info: {ssh_info}")
            
            if not parsed_ssh_info:
                print("❌ Could not extract SSH info from detailed job output")
                print("🔧 Trying alternative approach - checking stderr for complete output...")
                
                # Sometimes the full output is in stderr due to encoding issues
                # Try to find SSH login info in stderr even if it has encoding errors
                stderr_lines = result.stderr.split('\n')
                for line in stderr_lines:
                    if 'SSH login::' in line or 'ssh -i' in line:
                        print(f"🎯 Found potential SSH line in stderr: {line}")
                        import re
                        ssh_match = re.search(r"ssh -i '[^']*' ([A-Za-z0-9]+)@([a-z0-9.]+)", line)
                        if ssh_match:
                            container_id = ssh_match.group(1)
                            ssh_host = ssh_match.group(2)
                            job_info['container'] = container_id
                            job_info['ssh_host'] = ssh_host
                            print(f"🎯 Extracted container ID from stderr: '{container_id}'")
                            print(f"🎯 Extracted SSH host from stderr: '{ssh_host}'")
                            parsed_ssh_info = True
                            break
            # If we still don't have container info, try to derive it from job_id
            if 'container' not in job_info and job_info.get('status') in ['RUNNING', 'FINISHED']:
                print("🔍 No container found in output, trying to derive from job ID...")
                # GPULab often uses the first 8 characters of job ID as container identifier
                derived_container = job_id[:8]
                job_info['container'] = derived_container
                # SSH host should be extracted from detailed job info - if not available, connection will fail
                print(f"🎯 Using derived container ID: '{derived_container}'")
                print("⚠️ Warning: SSH host not determined - connection may fail")
            
            print(f"🔍 Final parsed job_info: {job_info}")
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
            print("💡 This saves time - no need to wait for the 30-minute download window to expire")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not cancel job - {e}")
            print("💡 Job will continue running for 30 minutes (download window)")
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
    
    def download_via_scp(self, job_id):
        """Download files using SCP with proper SSH ProxyCommand."""
        
        print(f"=== Downloading Files from Job {job_id[:8]} ===")
        
        # Get job info to extract container details
        job_info = self.get_job_info(job_id)
        
        if 'container' not in job_info or 'ssh_host' not in job_info:
            print("❌ Could not extract container information from job")
            print("Available job info:", job_info)
            return False
        
        container = job_info['container']
        ssh_host = job_info['ssh_host']        
        print(f"📡 Container: {container}")
        print(f"🖥️  SSH Host: {ssh_host}")
        
        # Set up local download directory
        local_lookup_dir = Path(self.download_dir) / "lookup_tables"
        local_lookup_dir.mkdir(parents=True, exist_ok=True)        
          # Generate the date-based remote directory path
        from datetime import datetime
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
        
        # Debug: Print the exact SSH command being used
        self.print_ssh_command_debug(ssh_check_cmd)
        
        try:
            result = subprocess.run(ssh_check_cmd, capture_output=True, text=True, timeout=45)
            
            print(f"🔍 SSH command output:")
            print(f"STDOUT: {repr(result.stdout)}")
            print(f"STDERR: {repr(result.stderr)}")
            print(f"Return code: {result.returncode}")
            
            # Check for SSH connection errors first
            if result.returncode != 0:
                if "Permission denied" in result.stderr or "Permission denied" in result.stdout:
                    print("🔐 SSH Permission Error Detected!")
                    print("❌ Cannot connect to remote server - SSH key authentication failed")
                    return False
                elif "Connection" in result.stderr or "timeout" in result.stderr.lower():
                    print("🌐 SSH Connection Error Detected!")
                    print("❌ Cannot reach remote server - network/connection issue")
                    return False
                elif "No such file or directory" in result.stdout:
                    print("📁 Remote directory doesn't exist or no pickle files found")                    # Try to find files in alternative locations
                    print("🔍 Searching for pickle files in alternative locations...")
                    search_cmd = ssh_check_cmd[:-1] + ["find /project_ghent/rsegawa/URENIMOD-lookups -name '*.pkl' 2>/dev/null | head -5"]
                    search_result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=30)
                    if search_result.returncode == 0 and search_result.stdout.strip():
                        print(f"� Found pickle files elsewhere:")
                        print(search_result.stdout)
                        # Use the first file found for download
                        first_file = search_result.stdout.strip().split('\n')[0]
                        pkl_files = [first_file]
                        print(f"� Will download: {first_file}")
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
                    "No pkl files found",  # Our custom message
                    "No such file or directory",  # Standard shell error
                    "cannot access",  # Another common error
                ]                
                has_pkl_files = (stdout_clean and 
                               not any(indicator in result.stdout for indicator in no_files_indicators) and
                               ".pkl" in result.stdout)
                
                if not has_pkl_files:
                    print("❌ No pickle files detected in expected directory")                    # Try alternative search
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
                            if full_path.strip():                                # Store as tuple: (full_path, filename)
                                filename = full_path.split('/')[-1]
                                pkl_files.append((full_path, filename))
                                print(f"📥 Will download: {filename} from {full_path}")
                    else:
                        print("❌ No pickle files found anywhere")
                        return False
                else:
                    print("📁 Available files:")
                    print(result.stdout)
                      # Extract filenames from ls output (these are regular files, just filenames)
                    pkl_files = []
                    for line in result.stdout.split('\n'):
                        if '.pkl' in line and not line.startswith('total'):
                            # Extract filename from ls -la output
                            parts = line.split()
                            if len(parts) >= 9:
                                filename = ' '.join(parts[8:])  # Handle filenames with spaces
                                # Just store the filename, not the full path since we already have remote_dir
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
            
            scp_cmd = [                "scp",
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
              # List all downloaded files
            print("\n📋 Downloaded lookup tables:")
            for pkl_file in local_lookup_dir.glob("*.pkl"):
                size = pkl_file.stat().st_size
                mod_time = datetime.fromtimestamp(pkl_file.stat().st_mtime)
                print(f"   📦 {pkl_file.name} ({size} bytes, {mod_time.strftime('%Y-%m-%d %H:%M')})")
            
            return True
        else:
            print("❌ No files were successfully downloaded")
            return False
    
    def run_complete_workflow(self):
        """Run the complete workflow: submit job, wait for completion, download files."""
        
        print("🚀 Starting URENIMOD Lookup Table Auto-Generation and Download")
        print("=" * 60)
        
        # Step 1: Submit job
        job_id = self.submit_job()
        if not job_id:
            print("❌ Workflow failed at job submission")
            return False
          # Step 2: Wait for completion (computation typically takes 3-5 minutes)
        completed = self.wait_for_completion(job_id, timeout_minutes=15)
        if not completed:
            print("❌ Workflow failed - job did not complete successfully")
            print("💡 You can still try to download manually if job is running:")
            print(f"   gpulab-cli --cert \"{self.cert_path}\" ssh {job_id[:8]}")
            return False
          # Step 3: Download files
        print("\n" + "=" * 40)
        downloaded = self.download_via_scp(job_id)
        
        if downloaded:
            print("\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            print("✅ Lookup tables generated and downloaded")
            print(f"📁 Files available at: {self.download_dir}")
            
            # Cancel the job to avoid waiting for the 30-minute timeout            print("\n" + "=" * 40)
            cancelled = self.cancel_job(job_id)
            if cancelled:
                print("✅ Job cancelled - no need to wait for timeout!")
            else:
                print("⚠️  Job continues running for 30 minutes (download window)")
            
            print("\n💡 You can now use these lookup tables in your URENIMOD research!")
            return True
        else:
            print("\n⚠️  Job completed but download failed")
            
            # Run diagnostics to help troubleshoot the issue
            self.diagnose_download_issues(job_id)
            
            print("\n💡 Manual download options:")
            print(f"1. SSH: gpulab-cli --cert \"{self.cert_path}\" ssh {job_id[:8]}")
            print(f"2. Check logs: gpulab-cli --cert \"{self.cert_path}\" log {job_id[:8]}")
            return False

    def test_ssh_connection(self, container, ssh_host):
        """Test SSH connectivity to the container before attempting file operations."""
        
        print("🔧 Testing SSH connection...")
        
        ssh_test_cmd = [
            "ssh", 
            "-i", self.cert_path,
            "-o", f"ProxyCommand=ssh -i {self.cert_path} fffrsegawau@bastion.ilabt.imec.be -W %h:%p",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{container}@{ssh_host}",
            "echo 'SSH connection test successful'"
        ]
        
        try:
            result = subprocess.run(ssh_test_cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and "SSH connection test successful" in result.stdout:
                print("✅ SSH connection test passed")
                return True
            else:
                print("❌ SSH connection test failed")
                print(f"   Return code: {result.returncode}")
                print(f"   STDOUT: {repr(result.stdout)}")
                print(f"   STDERR: {repr(result.stderr)}")
                
                # Provide specific guidance based on error type
                if "Permission denied" in result.stderr:
                    print("\n🔐 SSH Authentication Issue:")
                    print("   - Check if your SSH key file exists and has correct permissions")
                    print(f"   - Key path: {self.cert_path}")
                    print("   - The key should be readable only by you (600 permissions on Unix)")
                elif "Connection" in result.stderr or "timeout" in result.stderr.lower():
                    print("\n🌐 Network/Connection Issue:")
                    print("   - Check your internet connection")
                    print("   - The container might not be running anymore")
                    print("   - Try running the job status command to verify")
                elif "Host key verification failed" in result.stderr:
                    print("\n🔑 Host Key Issue:")
                    print("   - You may need to accept the host key manually first")
                    print("   - Try connecting manually once to accept the key")
                
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ SSH connection test timed out")
            print("   This could indicate network issues or the container being unresponsive")
            return False
        except Exception as e:
            print(f"❌ SSH connection test failed with error: {e}")
            return False

    def diagnose_download_issues(self, job_id):
        """Diagnose potential issues preventing successful file downloads."""
        
        print("\n🔍 Running diagnostic checks...")
        
        # Check 1: Verify job is still running/accessible
        print("1️⃣ Checking job status...")
        job_info = self.get_job_info(job_id)
        
        if not job_info:
            print("❌ Cannot retrieve job information - job may have been terminated")
            return
        
        status = job_info.get('status', 'UNKNOWN')
        print(f"   Job Status: {status}")
        
        if status not in ['RUNNING', 'COMPLETED']:
            print(f"⚠️  Job is in '{status}' state - this may prevent SSH access")
            print("💡 Consider rerunning the job if files are needed")
            return
          # Check 2: Test basic connectivity
        if 'container' in job_info and 'ssh_host' in job_info:
            container = job_info['container']
            ssh_host = job_info['ssh_host']
            print("2️⃣ Testing SSH connectivity...")
            
            if self.test_ssh_connection(container, ssh_host):
                print("✅ SSH connectivity is working")
                  # Check 3: Verify remote directory structure
                print("3️⃣ Checking remote directory structure...")
                
                ssh_cmd = [
                    "ssh", 
                    "-i", self.cert_path,
                    "-o", f"ProxyCommand=ssh -i {self.cert_path} fffrsegawau@bastion.ilabt.imec.be -W %h:%p",
                    "-o", "StrictHostKeyChecking=no",
                    f"{container}@{ssh_host}",
                    "find /project_ghent/rsegawa -type d -name '*URENIMOD*' 2>/dev/null | head -5"
                ]
                
                try:
                    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
                    if result.stdout.strip():
                        print("✅ Found URENIMOD directories:")
                        print(result.stdout)
                    else:
                        print("⚠️  No URENIMOD directories found in /project_ghent/rsegawa/")
                        print("💡 The computation may not have completed or files may be elsewhere")
                except Exception as e:
                    print(f"❌ Directory check failed: {e}")
            else:
                print("❌ SSH connectivity issues detected - see above for troubleshooting")
        
        print("\n💡 If issues persist, try:")
        print(f"   • Manual SSH: gpulab-cli --cert \"{self.cert_path}\" ssh {job_id[:8]}")
        print(f"   • Check logs: gpulab-cli --cert \"{self.cert_path}\" log {job_id[:8]}")
        print("   • Verify the job has completed computation (not just running)")
    
    def print_ssh_command_debug(self, ssh_cmd):
        """Print the exact SSH command for debugging purposes."""
        print("🔧 DEBUG: SSH Command being executed:")
        print("   " + " ".join(f'"{arg}"' if " " in arg else arg for arg in ssh_cmd))
        print()

def main():
    """Main function to run the auto-downloader."""
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("=== URENIMOD Lookup Table Auto-Generator ===")
    print(f"Working directory: {script_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if job file exists
    if not Path("gpulab_lookup_with_auto_download.json").exists():
        print("❌ Job file 'gpulab_lookup_with_auto_download.json' not found!")
        print("Please run this script from the GPULab directory.")
        return
    
    # Initialize and run
    downloader = GPULabAutoDownloader()
    success = downloader.run_complete_workflow()
    
    if success:
        print("\n🎊 All done! Happy researching with URENIMOD!")
    else:
        print("\n💔 Workflow had issues, but don't worry - we can troubleshoot!")

def test_container_extraction():
    """Test function to debug container ID extraction."""
    downloader = GPULabAutoDownloader()
    
    print("🧪 Testing container ID extraction...")
    
    # Test with the job that has container KD7GYH5I
    job_id = "a32cda61-8a62-44b5-bdda-3c2c6aabf5f8"
    print(f"Testing with job ID: {job_id}")
    
    job_info = downloader.get_job_info(job_id)
    print(f"Result: {job_info}")
    
    if 'container' in job_info and 'ssh_host' in job_info:
        print(f"✅ SUCCESS: Container: {job_info['container']}, SSH Host: {job_info['ssh_host']}")
        
        # Test the SSH command construction
        container = job_info['container']
        ssh_host = job_info['ssh_host']
        test_cmd = [
            "ssh", 
            "-i", downloader.cert_path,
            "-o", f"ProxyCommand=ssh -i {downloader.cert_path} fffrsegawau@bastion.ilabt.imec.be -W %h:%p",
            "-o", "StrictHostKeyChecking=no",
            f"{container}@{ssh_host}",
            "echo 'SSH test successful'"
        ]
        downloader.print_ssh_command_debug(test_cmd)
    else:
        print(f"❌ FAILED: Missing container or SSH host info")
        print(f"   Expected container: KD7GYH5I")
        print(f"   Expected SSH host: 4c.gpulab.ilabt.imec.be")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_container_extraction()
        elif sys.argv[1] == "--download-only":
            # Download-only mode: find the most recent running job and try to download
            downloader = GPULabAutoDownloader()
            
            # Get the most recent lookup job
            cert_path = downloader.cert_path
            project = downloader.project
            
            cmd = f'gpulab-cli --cert "{cert_path}" jobs'
            
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                                      encoding='utf-8', errors='replace', timeout=30)
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    lookup_jobs = []
                    
                    for line in lines:
                        if 'lookup_table' in line and 'rsegawau@ila' in line:
                            parts = line.split()
                            if len(parts) >= 8:
                                job_id = parts[0]
                                name = parts[1]
                                status = parts[-1]
                                lookup_jobs.append((job_id, name, status))
                    
                    if lookup_jobs:
                        # Use the most recent job
                        job_id, name, status = lookup_jobs[0]
                        print(f"🎯 Attempting download from job: {job_id[:8]} ({name})")
                        print(f"   Status: {status}")
                        
                        success = downloader.download_via_scp(job_id)
                        if success:
                            print("🎉 Download completed successfully!")
                        else:
                            print("❌ Download failed")
                    else:
                        print("❌ No lookup jobs found")
                else:
                    print("❌ Could not get job list")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("Usage: python gpulab_lookup_download_singlejob.py [--download-only | test]")
    else:
        main()

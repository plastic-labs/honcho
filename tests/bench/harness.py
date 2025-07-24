"""
Honcho Development Harness

A script that orchestrates running Honcho with a Docker database for development and testing.
This script:
1. Starts a PostgreSQL database in Docker with a configurable port
2. Provisions the database using Alembic migrations
3. Configures Honcho to use the database via environment variables
4. Starts the FastAPI server and deriver in separate processes
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml


class HonchoHarness:
    """
    Orchestrates running Honcho with a Docker database for development.
    """

    def __init__(self, db_port: int, project_root: Path) -> None:
        """
        Initialize the harness with database port and project root.

        Args:
            db_port: Port for the PostgreSQL database
            project_root: Path to the Honcho project root
        """
        self.db_port: int = db_port
        self.project_root: Path = project_root
        self.temp_dir: Path | None = None
        self.docker_compose_file: Path | None = None
        self.processes: list[tuple[str, subprocess.Popen[str]]] = []
        self.env_file_backup: Path | None = None
        self.output_threads: list[threading.Thread] = []

        # Set environment variables in the current process
        # This ensures they're inherited by all subprocesses
        for key, value in self.get_database_env_vars().items():
            os.environ[key] = value

    def create_temp_docker_compose(self) -> Path:
        """
        Create a temporary docker-compose.yml with the specified database port.

        Returns:
            Path to the temporary docker-compose.yml file
        """
        # Read the example docker-compose file
        example_file = self.project_root / "docker-compose.yml.example"
        with open(example_file) as f:
            compose_data = yaml.safe_load(f)

        # Update the database port
        compose_data["services"]["database"]["ports"] = [f"{self.db_port}:5432"]

        # Add a unique project name to avoid conflicts
        compose_data["name"] = f"honcho_harness_{self.db_port}"

        # Create temporary file
        self.temp_dir = Path(tempfile.mkdtemp(prefix="honcho_harness_"))
        self.docker_compose_file = self.temp_dir / "docker-compose.yml"

        with open(self.docker_compose_file, "w") as f:
            yaml.dump(compose_data, f, default_flow_style=False)

        print(f"Created temporary docker-compose.yml at {self.docker_compose_file}")
        return self.docker_compose_file

    def backup_env_file(self) -> None:
        """
        Temporarily rename the .env file to prevent it from overriding our environment variables.
        """
        env_file = self.project_root / ".env"
        if env_file.exists():
            backup_name = f".env.backup.{int(time.time())}"
            self.env_file_backup = env_file.rename(env_file.parent / backup_name)
            print(f"Temporarily renamed .env to {backup_name}")
        else:
            print("No .env file found, using environment variables only")

    def restore_env_file(self) -> None:
        """
        Restore the .env file from backup.
        """
        if self.env_file_backup:
            try:
                self.env_file_backup.rename(self.project_root / ".env")
                print("Restored .env file")
            except Exception as e:
                print(f"Error restoring .env file: {e}")

    def get_database_env_vars(self) -> dict[str, str]:
        """
        Get environment variables for database configuration and required API keys.

        Returns:
            Dictionary of environment variables for database connection and API keys
        """
        return {
            "DB_CONNECTION_URI": f"postgresql+psycopg://testuser:testpwd@localhost:{self.db_port}/honcho",
        }

    def start_database(self) -> None:
        """
        Start the PostgreSQL database using Docker Compose.
        """
        print(f"Starting PostgreSQL database on port {self.db_port}...")

        # Change to the temp directory and start the database service
        result = subprocess.run(
            [
                "docker-compose",
                "-f",
                str(self.docker_compose_file),
                "-p",
                f"honcho_harness_{self.db_port}",
                "up",
                "-d",
                "database",
            ],
            cwd=self.temp_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Failed to start database: {result.stderr}")
            sys.exit(1)

        print("Database started successfully")

    def wait_for_database(self, timeout: int = 60) -> bool:
        """
        Wait for the database to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if database is ready, False otherwise
        """
        print("Waiting for database to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Test database connection using pg_isready
                result = subprocess.run(
                    [
                        "pg_isready",
                        "-h",
                        "localhost",
                        "-p",
                        str(self.db_port),
                        "-U",
                        "testuser",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    print("Database is ready!")
                    return True

            except subprocess.TimeoutExpired:
                pass
            except FileNotFoundError:
                # pg_isready not available, try a different approach
                try:
                    import psycopg

                    conn = psycopg.connect(
                        f"postgresql://testuser:testpwd@localhost:{self.db_port}/honcho"
                    )
                    conn.close()
                    print("Database is ready!")
                    return True
                except Exception:
                    pass

            time.sleep(1)  # Faster polling for quicker startup

        print("Database failed to become ready within timeout")
        return False

    def provision_database(self) -> None:
        """
        Provision the database using the provision_db.py script.
        """
        print("Provisioning database...")

        # Run the provision script
        provision_script = self.project_root / "scripts" / "provision_db.py"
        result = subprocess.run(
            [sys.executable, str(provision_script)],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Failed to provision database: {result.stderr}")
            sys.exit(1)

        print("Database provisioned successfully")

    def verify_empty_database(self) -> None:
        """
        Verify that the database is empty with no workspaces and an empty queue.
        """

        try:
            import psycopg

            # Connect to the database
            conn_string = (
                f"postgresql://testuser:testpwd@localhost:{self.db_port}/honcho"
            )
            conn = psycopg.connect(conn_string)

            with conn.cursor() as cursor:
                # Check the workspaces table
                cursor.execute("SELECT COUNT(*) FROM workspaces")
                workspace_result = cursor.fetchone()
                workspace_count = workspace_result[0] if workspace_result else 0

                cursor.execute("SELECT COUNT(*) FROM queue")
                queue_result = cursor.fetchone()
                queue_count = queue_result[0] if queue_result else 0

            conn.close()

            # Report results
            if workspace_count != 0 or queue_count != 0:
                print("❌ Database verification failed: Database is not empty")
                print(
                    "This may indicate an issue with the database provisioning or cleanup."
                )
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error verifying database: {e}")
            print("Unable to verify database state. Continuing anyway...")

    def start_fastapi_server(self) -> subprocess.Popen[str]:
        """
        Start the FastAPI server.

        Returns:
            Process object for the FastAPI server
        """
        print("Starting FastAPI server...")

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--no-access-log",
                "--workers",
                "1",
            ],
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,
            universal_newlines=True,
        )

        self.processes.append(("FastAPI Server", process))
        return process

    def start_deriver(self) -> subprocess.Popen[str]:
        """
        Start the deriver process with debug logging enabled.

        Returns:
            Process object for the deriver
        """
        print("Starting deriver...")

        process = subprocess.Popen(
            [sys.executable, "-m", "src.deriver"],
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,
            universal_newlines=True,
        )

        self.processes.append(("Deriver", process))
        return process

    def stream_process_output(self, name: str, process: subprocess.Popen[str]) -> None:
        """
        Stream output from a process to stdout in real-time.

        Args:
            name: Name of the process for logging
            process: The subprocess to monitor
        """
        if not process.stdout:
            return

        # Patterns to filter out (noisy logs)
        filter_patterns = [
            "httpx -",
            # "src.routers.",
            # "src.crud.",
            "google_genai.models",
        ]

        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    line_stripped = line.rstrip()

                    # Skip lines that match filter patterns
                    if any(pattern in line_stripped for pattern in filter_patterns):
                        continue

                    print(f"[{name}] {line_stripped}")
        except Exception as e:
            print(f"Error reading from {name}: {e}")
        finally:
            if process.stdout:
                process.stdout.close()

    def wait_for_fastapi(self, timeout: int = 20) -> bool:
        """
        Wait for the FastAPI server to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if server is ready, False otherwise
        """
        print("Waiting for FastAPI server to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                import requests

                response = requests.get("http://localhost:8000/docs", timeout=5)
                if response.status_code == 200:
                    print("FastAPI server is ready!")
                    return True
            except Exception:
                pass

            # Check if the process has died and show output
            for name, process in self.processes:
                if process.poll() is not None:
                    print(f"❌ {name} process has died unexpectedly")
                    if process.stdout:
                        # Read all available output
                        output = process.stdout.read()
                        if output:
                            print(f"📋 {name} output:")
                            print(output)
                    return False

            time.sleep(1)  # Faster polling for quicker startup

        print("FastAPI server failed to become ready within timeout")

        # Show any available output from the FastAPI process
        for name, process in self.processes:
            if "FastAPI" in name and process.stdout:
                # Read any available output
                output = process.stdout.read()
                if output:
                    print(f"📋 {name} output:")
                    print(output)

        return False

    def print_honcho_config(self) -> None:
        """
        Fetch and print the actual configuration that Honcho is using.
        """
        print("\n" + "=" * 60)
        print("🔧 Honcho Configuration")
        print("=" * 60)

        # Create a Python script to import and print the settings
        config_script = f"""
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path("{self.project_root}")
sys.path.insert(0, str(project_root))

# Environment variables are already set in the parent process
# and will be inherited by this subprocess

try:
    from src.config import settings

    # Function to recursively print settings
    def print_settings(obj, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
            
        if hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):
                    full_key = f"{{prefix}}.{{key}}" if prefix else key
                    
                    # Handle nested settings objects
                    if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, type(None))):
                        print(f"\\n📋 {{full_key}}:")
                        print_settings(value, full_key, max_depth, current_depth + 1)
                    else:
                        # Mask sensitive information
                        if isinstance(value, str) and any(sensitive in value.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                            if 'testpwd' in value:
                                masked_value = value.replace('testpwd', '***')
                            else:
                                masked_value = '***'
                        else:
                            masked_value = value
                        print(f"  {{key}}: {{masked_value}}")
    
    # Print all settings
    print_settings(settings)
    
except Exception as e:
    print(f"❌ Error importing settings: {{e}}")
    import traceback
    traceback.print_exc()
"""

        # Write the script to a temporary file
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="honcho_harness_"))

        script_file = self.temp_dir / "print_config.py"
        with open(script_file, "w") as f:
            f.write(config_script)

        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_file)],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Error running config script: {result.stderr}")

        print("=" * 60)

    def cleanup(self) -> None:
        """
        Clean up resources and stop all processes.
        """
        print("\nCleaning up...")

        # Stop all processes
        for name, process in self.processes:
            print(f"Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"Error stopping {name}: {e}")

        # Wait for output threads to finish
        for thread in self.output_threads:
            if thread.is_alive():
                thread.join(timeout=2)

        # Stop database
        if self.docker_compose_file and self.docker_compose_file.exists():
            print("Stopping database...")
            try:
                # More aggressive cleanup - remove containers, volumes, and orphaned containers
                subprocess.run(
                    [
                        "docker-compose",
                        "-f",
                        str(self.docker_compose_file),
                        "-p",
                        f"honcho_harness_{self.db_port}",
                        "down",
                        "--volumes",
                        "--remove-orphans",
                    ],
                    cwd=self.temp_dir,
                    capture_output=True,
                    text=True,
                )

                # Also try to remove any containers that might still be running
                subprocess.run(
                    ["docker", "ps", "-q", "--filter", "name=honcho_harness"],
                    capture_output=True,
                    text=True,
                )

            except Exception as e:
                print(f"Error stopping database: {e}")

        # Remove temporary files
        if self.temp_dir and self.temp_dir.exists():
            try:
                import shutil

                shutil.rmtree(self.temp_dir)
                print(f"Removed temporary directory {self.temp_dir}")
            except Exception as e:
                print(f"Error removing temp directory: {e}")

        # Restore .env file
        self.restore_env_file()

    def run(self) -> None:
        """
        Run the complete Honcho harness.
        """
        try:
            # Backup .env file to prevent it from overriding our environment variables
            self.backup_env_file()

            # Copy .env file from tests/bench to the project root
            shutil.copy(
                self.project_root / "tests" / "bench" / ".env", self.project_root
            )

            # Create temporary docker-compose.yml
            self.create_temp_docker_compose()

            # Start database
            self.start_database()

            # Wait for database to be ready
            if not self.wait_for_database():
                print("Database failed to start. Exiting.")
                sys.exit(1)

            # Provision database
            self.provision_database()

            # Verify database is empty
            self.verify_empty_database()

            # Start FastAPI server
            _fastapi_process = self.start_fastapi_server()

            # Wait for FastAPI to be ready
            if not self.wait_for_fastapi():
                print("FastAPI server failed to start. Exiting.")
                sys.exit(1)

            # Print the actual configuration Honcho is using
            self.print_honcho_config()

            # Start deriver
            _deriver_process = self.start_deriver()

            print("\n" + "=" * 60)
            print("🎉 Honcho is running!")
            print(f"📊 Database: localhost:{self.db_port}")
            print("🌐 API Server: http://localhost:8000")
            print("📚 API Docs: http://localhost:8000/docs")
            print("🔄 Deriver: Running")
            print("=" * 60)
            print("Press Ctrl+C to stop all services")
            print("=" * 60 + "\n")

            # Start output streaming threads for each process
            for name, process in self.processes:
                thread = threading.Thread(
                    target=self.stream_process_output, args=(name, process), daemon=True
                )
                thread.start()
                self.output_threads.append(thread)

            # Monitor processes for unexpected termination
            while True:
                # Check if any process has died
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"❌ {name} has stopped unexpectedly")
                        return

                time.sleep(1)  # Check every second

        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()


def main():
    """
    Main entry point for the Honcho harness.
    """
    parser = argparse.ArgumentParser(
        description="Run Honcho with a Docker database for development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --port 5433                    # Run with database on port 5433
  %(prog)s --port 5434 --project-root /path/to/honcho  # Custom project root
        """,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5433,
        help="Port for the PostgreSQL database (default: 5433)",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the Honcho project root (default: current directory)",
    )

    args = parser.parse_args()

    # Validate project root
    if not (args.project_root / "src" / "main.py").exists():
        print(
            f"Error: {args.project_root} does not appear to be a valid Honcho project root"
        )
        print("Make sure you're running this script from the Honcho project directory")
        sys.exit(1)

    # Check for required files
    required_files = [
        "docker-compose.yml.example",
        "config.toml.example",
        "scripts/provision_db.py",
    ]

    for file_path in required_files:
        if not (args.project_root / file_path).exists():
            print(f"Error: Required file {file_path} not found in {args.project_root}")
            sys.exit(1)

    # Create and run the harness
    harness = HonchoHarness(args.port, args.project_root)
    harness.run()


if __name__ == "__main__":
    main()

# src/dpst/cli.py
import sys
import argparse
from dpst.setup_db import initialize_database, run_clustering_and_indexing

def main():
    parser = argparse.ArgumentParser(description="DPST Package Management Utility")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("setup", help="Initialize Weaviate DB, download FineWeb, and generate cluster JSON metadata files.")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        print("--- Starting DPST Setup Process ---")
        try:
            initialize_database()
            run_clustering_and_indexing()
            print("--- Setup Successful! You can now use DPST ---")
        except Exception as e:
            print(f"Setup failed with error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
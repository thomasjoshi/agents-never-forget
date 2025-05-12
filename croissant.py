import requests
import json
import argparse

def get_croissant_metadata(dataset_name, api_token=None):
    """
    Retrieve Croissant metadata from Hugging Face API for a given dataset.
    
    Args:
        dataset_name: Name of the dataset in format 'username/dataset-name'
        api_token: Optional Hugging Face API token for private datasets
        
    Returns:
        Dictionary containing the Croissant metadata or None if failed
    """
    # Set up headers with API token if provided
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    # Construct API URL
    API_URL = f"https://huggingface.co/api/datasets/{dataset_name}/croissant"
    
    response = requests.get(API_URL, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve metadata: {response.status_code}")
        print(response.text)
        return None

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Get Croissant metadata for a Hugging Face dataset")
    parser.add_argument("dataset", help="Dataset name in format 'username/dataset-name'")
    parser.add_argument("--token", help="Hugging Face API token for private datasets")
    parser.add_argument("--output", default="croissant.json", help="Output filename (default: croissant.json)")
    args = parser.parse_args()
    
    # Extract dataset name
    dataset_name = args.dataset
    if dataset_name.startswith("https://huggingface.co/datasets/"):
        dataset_name = dataset_name.replace("https://huggingface.co/datasets/", "")
    
    print(f"Fetching Croissant metadata for dataset: {dataset_name}")
    metadata = get_croissant_metadata(dataset_name, args.token)
    
    if metadata:
        # Save the metadata to the specified file
        with open(args.output, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Croissant metadata saved to {args.output}")
        print(f"You can validate this file using the Croissant validator before submitting.")
    else:
        print("Failed to retrieve Croissant metadata.")

if __name__ == "__main__":
    main()

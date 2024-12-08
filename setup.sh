#!/bin/bash

#Repos to checkout
repos=(
    "https://github.com/open-mmlab/mmcv.git"
    "https://github.com/ViTAE-Transformer/ViTPose.git"
    "https://github.com/shubham-goel/4D-Humans.git FDHumans"
)

# Directory to clone repositories into
clone_dir="./repos"

# Create the directory if it doesn't exist
mkdir -p "$clone_dir"

# Navigate to the directory
cd "$clone_dir" || { echo "Failed to enter directory $clone_dir"; exit 1; }

# Function to clone a repository and optionally rename it
clone_repo() {
    local repo_info="$1"
    local repo_url
    local custom_name

    # Split the repo_info into URL and custom name (if provided)
    repo_url=$(echo "$repo_info" | awk '{print $1}')
    custom_name=$(echo "$repo_info" | awk '{print $2}')

    # Extract the repo name from URL
    local repo_name
    repo_name=$(basename "$repo_url" .git)

    echo "Cloning $repo_name..."

    # Clone the repository
    if git clone "$repo_url"; then
        echo "Successfully cloned $repo_name."

        # Rename the repository directory if a custom name is provided
        if [[ -n "$custom_name" && "$custom_name" != "$repo_name" ]]; then
            mv "$repo_name" "$custom_name" && echo "Renamed $repo_name to $custom_name."
        fi
    else
        echo "Error cloning $repo_name. Skipping..."
    fi
}

# Loop through the repository list and clone each one
for repo in "${repos[@]}"; do
    clone_repo "$repo"
done

echo "All repositories processed."
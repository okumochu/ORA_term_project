import os
from gen_diff_topo import transform_to_FJSSP

def main():
    # Get the directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ta_dir = os.path.join(script_dir, "..", "ta")
    output_dir = os.path.join(script_dir, "output")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all .txt files from ta directory and sort them
    ta_files = [f for f in os.listdir(ta_dir) if f.endswith(".txt")]
    ta_files.sort()

    # Use the first 30 files
    first_30_files = ta_files[:30]

    # Process each file
    for filename in first_30_files:
        input_path = os.path.join(ta_dir, filename)
        base_name = os.path.splitext(filename)[0]  # e.g., "ta01"

        # Generate outputs for flex = 2, 3, 4 and closed, linked
        for flex in [2, 3, 4]:
            # Closed topology
            closed_output_path = os.path.join(output_dir, f"{base_name}_closed_flex{flex}.txt")
            transform_to_FJSSP(input_path, flex, closed_output_path, closed=True)
            print(f"Generated: {closed_output_path}")

            # Linked topology
            linked_output_path = os.path.join(output_dir, f"{base_name}_linked_flex{flex}.txt")
            transform_to_FJSSP(input_path, flex, linked_output_path, closed=False)
            print(f"Generated: {linked_output_path}")

    print(f"\nCompleted! Generated {len(first_30_files) * 6} files in {output_dir}")


if __name__ == "__main__":
    main()

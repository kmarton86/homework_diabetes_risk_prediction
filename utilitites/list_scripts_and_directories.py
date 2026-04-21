import os
from tkinter import Tk, filedialog

# File extensions to include in content dump (editable)
INCLUDE_EXTENSIONS = {
    ".py", ".txt", ".yml", ".yaml", ".json", ".ini", ".cfg", ".bat",
    ".sh", ".dockerfile", ".env", ".toml", ".md", ".js", ".ts", ".html",
    ".css", ".php", ".rb"
}

def list_directory_structure(base_dir):
    """Generate a directory + file tree overview."""
    lines = ["=== PROJECT STRUCTURE ===", ""]
    base_dir = os.path.abspath(base_dir)

    for root, dirs, files in os.walk(base_dir):
        rel_path = os.path.relpath(root, base_dir)
        folder_label = f"[{base_dir}]" if rel_path == "." else f"[{rel_path}]"
        lines.append(folder_label)
        if not files:
            lines.append("  (no files)")
        else:
            for f in sorted(files):
                lines.append(f"  - {f}")
        lines.append("")  # space between dirs

    return "\n".join(lines)

def read_file_content(path):
    """Read text file safely and return its contents."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content if content else "(empty file)"
    except Exception as e:
        return f"[Error reading file: {e}]"

def list_file_contents(base_dir):
    """Return formatted file contents for all text/script files."""
    lines = ["", "=== FILE CONTENTS ===", ""]
    base_dir = os.path.abspath(base_dir)

    for root, dirs, files in os.walk(base_dir):
        rel_path = os.path.relpath(root, base_dir)
        folder_label = f"[{base_dir}]" if rel_path == "." else f"[{rel_path}]"
        has_files = False

        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in INCLUDE_EXTENSIONS:
                if not has_files:
                    lines.append(folder_label)
                    has_files = True
                file_path = os.path.join(root, f)
                content = read_file_content(file_path)
                lines.append(f"  - {f}:")
                # Indent file contents
                indented = "\n".join("    " + line for line in content.splitlines())
                lines.append(indented + "\n")

    return "\n".join(lines)

def main():
    Tk().withdraw()
    directory = filedialog.askdirectory(title="Select a directory to summarize")
    if not directory:
        print("No directory selected. Exiting.")
        return

    print(f"📂 Scanning: {directory} ...")

    # Build sections
    structure = list_directory_structure(directory)
    contents = list_file_contents(directory)

    # Combine into one formatted output
    final_output = f"{structure}\n{contents}"
    output_path = os.path.join(directory, "project_summary.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"\n✅ Project summary saved to:\n{output_path}")

if __name__ == "__main__":
    main()

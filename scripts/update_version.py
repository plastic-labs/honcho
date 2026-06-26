#!/usr/bin/env uv run python
"""
Honcho Version Update Script

This script helps update version numbers across the Honcho repository.
It handles the main API, Python SDK, and TypeScript SDK in a single operation.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime


class VersionUpdater:
    def __init__(self, base_path: str):
        self.base_path: str = base_path

    def get_current_versions(self) -> dict[str, str]:
        """Get current version numbers from the repository."""
        versions: dict[str, str] = {}

        # Main API version
        with open(os.path.join(self.base_path, "pyproject.toml")) as f:
            for line in f:
                if line.startswith("version = "):
                    versions["api"] = line.split('"')[1]
                    break

        # Python SDK version
        with open(os.path.join(self.base_path, "sdks/python/pyproject.toml")) as f:
            for line in f:
                if line.startswith("version = "):
                    versions["python_sdk"] = line.split('"')[1]
                    break

        # TypeScript SDK version
        with open(os.path.join(self.base_path, "sdks/typescript/package.json")) as f:
            data = json.load(f)
            versions["typescript_sdk"] = data["version"]

        return versions

    def get_all_versions_from_editor(
        self, current_versions: dict[str, str]
    ) -> dict[str, dict[str, str]]:
        """Open editor to get all version updates at once."""
        template = f"""# Honcho Version Update
# Enter new version numbers below. Leave blank to skip updating that component.
#
# MAIN API
# Current version: {current_versions["api"]}
API_VERSION=

# API Changelog (use ### for section headers: Added, Changed, Fixed, etc.)


# PYTHON SDK
# Current version: {current_versions["python_sdk"]}
PYTHON_VERSION=

# Python SDK Changelog


# TYPESCRIPT SDK
# Current version: {current_versions["typescript_sdk"]}
TYPESCRIPT_VERSION=

# TypeScript SDK Changelog


# Lines starting with # are comments and will be ignored
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(template)
            temp_file = f.name

        # Open in vim
        subprocess.call(["vim", temp_file])

        # Parse the file
        with open(temp_file) as f:
            content = f.read()

        os.unlink(temp_file)

        # Extract all versions and changelogs
        updates: dict[str, dict[str, str]] = {}

        # Parse API version
        api_match = re.search(r"^API_VERSION=(.*)$", content, re.MULTILINE)
        if api_match and api_match.group(1).strip():
            changelog = self._extract_changelog_between(
                content, "API_VERSION=", "PYTHON_VERSION="
            )
            updates["api"] = {
                "version": api_match.group(1).strip(),
                "changelog": self._clean_changelog_sections(changelog),
            }

        # Parse Python SDK version
        python_match = re.search(r"^PYTHON_VERSION=(.*)$", content, re.MULTILINE)
        if python_match and python_match.group(1).strip():
            changelog = self._extract_changelog_between(
                content, "PYTHON_VERSION=", "TYPESCRIPT_VERSION="
            )
            updates["python_sdk"] = {
                "version": python_match.group(1).strip(),
                "changelog": self._clean_changelog_sections(changelog),
            }

        # Parse TypeScript SDK version
        ts_match = re.search(r"^TYPESCRIPT_VERSION=(.*)$", content, re.MULTILINE)
        if ts_match and ts_match.group(1).strip():
            changelog = self._extract_changelog_between(
                content, "TYPESCRIPT_VERSION=", None
            )
            updates["typescript_sdk"] = {
                "version": ts_match.group(1).strip(),
                "changelog": self._clean_changelog_sections(changelog),
            }

        return updates

    def _extract_changelog_between(
        self, content: str, start_marker: str, end_marker: str | None
    ) -> str:
        """Extract changelog content between markers."""
        lines = content.split("\n")
        changelog_lines: list[str] = []
        in_section = False

        for line in lines:
            if start_marker in line:
                in_section = True
                continue
            if end_marker and end_marker in line:
                break
            if in_section:
                # Skip comment lines but keep markdown headers
                if line.strip().startswith("# ") and not line.strip().startswith("###"):
                    continue
                if line.strip() == "#":
                    continue
                changelog_lines.append(line)

        # Remove trailing empty lines
        while changelog_lines and not changelog_lines[-1].strip():
            changelog_lines.pop()

        return "\n".join(changelog_lines).strip()

    def _clean_changelog_sections(self, changelog: str) -> str:
        """Remove empty changelog sections."""
        sections = ["Added", "Changed", "Fixed", "Deprecated", "Removed", "Security"]
        lines = changelog.split("\n")
        cleaned_lines: list[str] = []
        current_section: str | None = None
        section_has_content = False
        section_start_idx = -1
        for i, line in enumerate(lines):
            # Check if this is a section header
            is_section_header = False
            for section in sections:
                if line.strip() == f"### {section}":
                    # If we have a previous section, decide whether to keep it
                    if (
                        current_section is not None
                        and section_start_idx != -1
                        and section_has_content
                    ):
                        # Keep the section
                        cleaned_lines.extend(lines[section_start_idx:i])
                    # Start tracking new section
                    current_section = section
                    section_start_idx = i
                    section_has_content = False
                    is_section_header = True
                    break

            if (
                not is_section_header
                and current_section is not None
                and line.strip()
                and not line.strip().startswith("#")
            ):
                section_has_content = True

        # Handle the last section
        if (
            current_section is not None
            and section_start_idx != -1
            and section_has_content
        ):
            cleaned_lines.extend(lines[section_start_idx:])

        # If no sections were found, return original
        if not cleaned_lines and "###" not in changelog:
            return changelog

        return "\n".join(cleaned_lines).strip()

    def update_all(
        self, updates: dict[str, dict[str, str]], current_versions: dict[str, str]
    ):
        """Update all components that have new versions."""
        # Update API if specified
        if "api" in updates:
            print(f"\nUpdating API version to {updates['api']['version']}...")
            self.update_api_version(
                updates["api"]["version"], updates["api"]["changelog"]
            )

            # Update compatibility guide for new API version
            self._update_compatibility_guide_for_api(
                updates["api"]["version"],
                updates.get("python_sdk", {}).get(
                    "version", current_versions["python_sdk"]
                ),
                updates.get("typescript_sdk", {}).get(
                    "version", current_versions["typescript_sdk"]
                ),
            )

        # Update Python SDK if specified
        if "python_sdk" in updates:
            print(
                f"Updating Python SDK version to {updates['python_sdk']['version']}..."
            )
            self.update_python_sdk_version(
                updates["python_sdk"]["version"], updates["python_sdk"]["changelog"]
            )

        # Update TypeScript SDK if specified
        if "typescript_sdk" in updates:
            print(
                f"Updating TypeScript SDK version to {updates['typescript_sdk']['version']}..."
            )
            self.update_typescript_sdk_version(
                updates["typescript_sdk"]["version"],
                updates["typescript_sdk"]["changelog"],
            )

    def update_api_version(self, new_version: str, changelog: str):
        """Update main API version across all files."""
        updates = [
            # Simple replacements
            {
                "file": "pyproject.toml",
                "pattern": r'version = "[^"]*"',
                "replacement": f'version = "{new_version}"',
            },
            {
                "file": "src/main.py",
                "pattern": r'version="[^"]*"',
                "replacement": f'version="{new_version}"',
            },
            {
                "file": "README.md",
                "pattern": r"Version-\d+\.\d+\.\d+-blue",
                "replacement": f"Version-{new_version}-blue",
            },
            # docs.json is handled separately to only update same major version
        ]

        # Apply simple updates
        for update in updates:
            file_path = os.path.join(self.base_path, update["file"])
            with open(file_path) as f:
                content = f.read()

            content = re.sub(update["pattern"], update["replacement"], content)

            with open(file_path, "w") as f:
                f.write(content)

        # Update docs.json - only update same major version
        self._update_docs_json(new_version)

        # Update CHANGELOG.md (prepend new entry)
        self._update_changelog_md(new_version, changelog)

        # Update docs changelog (MDX format)
        self._update_docs_changelog(new_version, changelog, "api")

    def update_python_sdk_version(self, new_version: str, changelog: str):
        """Update Python SDK version."""
        updates = [
            {
                "file": "sdks/python/pyproject.toml",
                "pattern": r'version = "[^"]*"',
                "replacement": f'version = "{new_version}"',
            },
            {
                "file": "sdks/python/src/honcho/__init__.py",
                "pattern": r'__version__ = "[^"]*"',
                "replacement": f'__version__ = "{new_version}"',
            },
        ]

        for update in updates:
            file_path = os.path.join(self.base_path, update["file"])
            with open(file_path) as f:
                content = f.read()

            content = re.sub(update["pattern"], update["replacement"], content)

            with open(file_path, "w") as f:
                f.write(content)

        # Update SDK's own CHANGELOG.md
        self._update_sdk_changelog(new_version, changelog, "sdks/python/CHANGELOG.md")

        # Update docs changelog
        self._update_docs_changelog(new_version, changelog, "python_sdk")

        # Update compatibility guide SDK version
        self._update_compatibility_guide("python", new_version)

    def update_typescript_sdk_version(self, new_version: str, changelog: str):
        """Update TypeScript SDK version."""
        # Update package.json
        file_path = os.path.join(self.base_path, "sdks/typescript/package.json")
        with open(file_path) as f:
            data = json.load(f)

        data["version"] = new_version

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")  # Add trailing newline

        # Update SDK's own CHANGELOG.md
        self._update_sdk_changelog(
            new_version, changelog, "sdks/typescript/CHANGELOG.md"
        )

        # Update docs changelog
        self._update_docs_changelog(new_version, changelog, "typescript_sdk")

        # Update compatibility guide SDK version
        self._update_compatibility_guide("typescript", new_version)

    def _update_docs_json(self, new_version: str):
        """Update docs.json version label(s) sharing the new version's major.

        Uses a targeted regex replacement rather than a JSON round-trip so the
        file's existing formatting (compact inline arrays) is preserved instead
        of being reflowed.
        """
        file_path = os.path.join(self.base_path, "docs/docs.json")

        with open(file_path) as f:
            content = f.read()

        # Get major version of new version
        new_major = new_version.split(".")[0]

        def _replace(match: re.Match[str]) -> str:
            # Only update labels whose major version matches the new version's.
            if match.group(1) == new_major:
                return f'"version": "v{new_version}"'
            return match.group(0)

        content = re.sub(r'"version": "v(\d+)\.\d+\.\d+"', _replace, content)

        with open(file_path, "w") as f:
            f.write(content)

    def _update_sdk_changelog(self, version: str, changelog: str, relative_path: str):
        """Update an SDK's CHANGELOG.md file."""
        file_path = os.path.join(self.base_path, relative_path)

        with open(file_path) as f:
            content = f.read()

        # Find the position after the header
        header_end = content.find("\n## [")
        if header_end == -1:
            header_end = content.find("\n##")

        if header_end == -1:
            # No existing entries, add after title section
            header_end = content.find("and this project adheres to")
            if header_end != -1:
                header_end = content.find("\n", header_end)

        # Create new entry with proper formatting
        date = datetime.now().strftime("%Y-%m-%d")

        # Ensure changelog content is properly formatted
        if changelog.strip():
            formatted_changelog = changelog.strip()
        else:
            formatted_changelog = "### Changed\n\n- Updated version"

        new_entry = f"\n\n## [{version}] - {date}\n\n{formatted_changelog}\n"

        # Insert the new entry
        new_content = content[:header_end] + new_entry + content[header_end:]

        with open(file_path, "w") as f:
            f.write(new_content)

    def _update_changelog_md(self, version: str, changelog: str):
        """Update the main CHANGELOG.md file.

        If an ``## [Unreleased]`` section is present, it is promoted to the new
        version (its contents replaced by ``changelog``, which the caller is
        expected to have already merged). Otherwise a new version entry is
        prepended above the most recent release, preserving the legacy behavior.
        """
        file_path = os.path.join(self.base_path, "CHANGELOG.md")

        with open(file_path) as f:
            content = f.read()

        date = datetime.now().strftime("%Y-%m-%d")

        # Ensure changelog content is properly formatted
        if changelog.strip():
            formatted_changelog = changelog.strip()
        else:
            formatted_changelog = "### Changed\n\n- Updated version"

        # Promote an existing [Unreleased] section if one exists. Match from the
        # "## [Unreleased]" header up to (but not including) the next release
        # heading, and replace the whole block with the new version section.
        unreleased_re = re.compile(
            r"\n## \[Unreleased\][\s\S]*?(?=\n## \[)", re.IGNORECASE
        )
        if unreleased_re.search(content):
            replacement = f"\n## [{version}] - {date}\n\n{formatted_changelog}\n"
            new_content = unreleased_re.sub(replacement, content, count=1)
            with open(file_path, "w") as f:
                f.write(new_content)
            return

        # Find the position after the header
        header_end = content.find("\n## [")
        if header_end == -1:
            header_end = content.find("\n##")

        if header_end == -1:
            # No existing entries, add after title
            header_end = content.find("\n", content.find("# Changelog"))

        new_entry = f"\n\n## [{version}] - {date}\n\n{formatted_changelog}\n"

        # Insert the new entry
        new_content = content[:header_end] + new_entry + content[header_end:]

        with open(file_path, "w") as f:
            f.write(new_content)

    def _update_docs_changelog(self, version: str, changelog: str, component: str):
        """Update the docs/changelog/introduction.mdx file."""
        file_path = os.path.join(self.base_path, "docs/changelog/introduction.mdx")

        with open(file_path) as f:
            content = f.read()

        if component == "api":
            # Find the Honcho API tab content
            tab_start = content.find('<Tab title="Honcho API">')
            if tab_start == -1:
                return

            # Find where to insert (after the Tab opening)
            insert_pos = content.find("\n", tab_start) + 1

            # Format the changelog with proper indentation
            indented_changelog = "\n".join(
                "        " + line if line.strip() else ""
                for line in changelog.strip().split("\n")
            )

            # Create the new update entry
            new_entry = f"""        <Update label="v{version} (Current)">
{indented_changelog}
        </Update>

"""

            # Remove (Current) from previous entries
            # Use a more specific pattern to avoid replacing in other contexts
            content = re.sub(
                r'(<Update label="v[^"]*) \(Current\)(")', r"\1\2", content
            )

            # Insert the new entry right after the Tab line
            content = content[:insert_pos] + new_entry + content[insert_pos:]

        elif component == "python_sdk":
            # Update Python SDK section
            content = self._update_sdk_changelog_section(
                content, "Python SDK", version, changelog
            )

        elif component == "typescript_sdk":
            # Update TypeScript SDK section
            content = self._update_sdk_changelog_section(
                content, "TypeScript SDK", version, changelog
            )

        with open(file_path, "w") as f:
            f.write(content)

    def _update_sdk_changelog_section(
        self, content: str, sdk_title: str, version: str, changelog: str
    ):
        """Update a specific SDK section in the changelog."""
        tab_pattern = f'<Tab title="{sdk_title}">'
        tab_start = content.find(tab_pattern)
        if tab_start == -1:
            return content

        # Format the changelog with proper indentation
        indented_changelog = "\n".join(
            "            " + line if line.strip() else ""
            for line in changelog.strip().split("\n")
        )

        # Create new update entry
        new_entry = f"""        <Update label="v{version} (Current)">
{indented_changelog}
        </Update>
"""

        # Remove (Current) from previous SDK entries
        # More precise pattern to avoid issues
        pattern = rf'(<Tab title="{sdk_title}">.*?<Update label="v[^"]*) \(Current\)(".*?</Tab>)'
        content = re.sub(pattern, r"\1\2", content, flags=re.DOTALL)

        # Find where to insert the new entry
        # Look for the line after the SDK link (e.g., [Python SDK](...))
        tab_pos = content.find(tab_pattern)
        if tab_pos == -1:
            return content

        # Find the end of the SDK link line
        link_start = content.find("[", tab_pos)
        if link_start != -1:
            link_end = content.find("\n", link_start)
            if link_end != -1:
                insert_pos = link_end + 1
                content = content[:insert_pos] + new_entry + content[insert_pos:]

        return content

    def _update_compatibility_guide(self, sdk_type: str, version: str):
        """Update the compatibility guide with new SDK version."""
        file_path = os.path.join(
            self.base_path, "docs/changelog/compatibility-guide.mdx"
        )

        with open(file_path) as f:
            content = f.read()

        if sdk_type == "typescript":
            # Update in the card
            content = re.sub(
                r'(<Card title="TypeScript SDK".*?Compatible Version:\*\*) v[\d.]+',
                rf"\1 v{version}",
                content,
                flags=re.DOTALL,
            )
            # Update in the install command
            content = re.sub(
                r"npm install @honcho-ai/sdk@[\d.]+",
                f"npm install @honcho-ai/sdk@{version}",
                content,
            )
        elif sdk_type == "python":
            # Update in the card
            content = re.sub(
                r'(<Card title="Python SDK".*?Compatible Version:\*\*) v[\d.]+',
                rf"\1 v{version}",
                content,
                flags=re.DOTALL,
            )
            # Update in the install command
            content = re.sub(
                r"pip install honcho-ai==[\d.]+",
                f"pip install honcho-ai=={version}",
                content,
            )

        with open(file_path, "w") as f:
            f.write(content)

    def _update_compatibility_guide_for_api(
        self, api_version: str, python_version: str, typescript_version: str
    ):
        """Update compatibility guide when API version changes."""
        file_path = os.path.join(
            self.base_path, "docs/changelog/compatibility-guide.mdx"
        )

        with open(file_path) as f:
            content = f.read()

        # Update the current API version header
        content = re.sub(
            r"### Honcho API v[\d.]+ \(Current\)",
            f"### Honcho API v{api_version} (Current)",
            content,
        )

        # Find the table and update/add entry
        table_match = re.search(
            r"(\| Honcho API Version.*?\n\|[-| ]+\n)(.*?)(\n\n|$)", content, re.DOTALL
        )
        if table_match:
            header = table_match.group(1)
            rows = table_match.group(2)
            after_table = table_match.group(3)

            # Update existing current version
            rows = re.sub(
                r"v[\d.]+ \(Current\)",
                lambda m: m.group(0).replace(" (Current)", ""),
                rows,
            )

            # Add new row at the top
            new_row = f"| v{api_version} (Current) | v{typescript_version} | v{python_version} |"

            # Reconstruct table
            new_table = header + new_row + "\n" + rows + after_table

            # Replace in content
            content = (
                content[: table_match.start()]
                + new_table
                + content[table_match.end() :]
            )

        with open(file_path, "w") as f:
            f.write(content)


def _resolve_changelog(value: str | None) -> str:
    """Resolve a changelog argument that is either inline text or a file path."""
    if not value:
        return ""
    if os.path.isfile(value):
        with open(value) as f:
            return f.read().strip()
    return value.strip()


def _updates_from_args(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """Build the updates dict from CLI flags (headless mode)."""
    updates: dict[str, dict[str, str]] = {}
    if args.api_version:
        updates["api"] = {
            "version": args.api_version,
            "changelog": _resolve_changelog(args.api_changelog),
        }
    if args.python_version:
        updates["python_sdk"] = {
            "version": args.python_version,
            "changelog": _resolve_changelog(args.python_changelog),
        }
    if args.typescript_version:
        updates["typescript_sdk"] = {
            "version": args.typescript_version,
            "changelog": _resolve_changelog(args.typescript_changelog),
        }
    return updates


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Update Honcho version numbers and changelogs. With no version "
            "flags, opens an interactive editor; pass one or more --*-version "
            "flags to run headless (agent-friendly)."
        )
    )
    parser.add_argument("--api-version", help="New Main API version.")
    parser.add_argument("--python-version", help="New Python SDK version.")
    parser.add_argument("--typescript-version", help="New TypeScript SDK version.")
    parser.add_argument(
        "--api-changelog",
        help="API changelog markdown, or a path to a file containing it.",
    )
    parser.add_argument(
        "--python-changelog",
        help="Python SDK changelog markdown, or a path to a file containing it.",
    )
    parser.add_argument(
        "--typescript-changelog",
        help="TypeScript SDK changelog markdown, or a path to a file containing it.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (implied in headless mode).",
    )
    args = parser.parse_args()

    # Get the parent directory of the scripts folder (the project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(script_dir)
    updater = VersionUpdater(base_path)

    # Get current versions
    current_versions = updater.get_current_versions()

    headless = any([args.api_version, args.python_version, args.typescript_version])

    print("Honcho Version Updater")
    print("=" * 50)
    print("\nCurrent versions:")
    print(f"  Main API:        {current_versions['api']}")
    print(f"  Python SDK:      {current_versions['python_sdk']}")
    print(f"  TypeScript SDK:  {current_versions['typescript_sdk']}")
    print()

    if headless:
        updates = _updates_from_args(args)
    else:
        print("Opening editor for version updates...")
        print("Leave version fields blank to skip updating that component.")
        print()
        # Get all updates at once
        updates = updater.get_all_versions_from_editor(current_versions)

    if not updates:
        print("No versions specified. Exiting...")
        sys.exit(0)

    # Show what will be updated
    print("\nThe following components will be updated:")
    for component, info in updates.items():
        component_name = {
            "api": "Main API",
            "python_sdk": "Python SDK",
            "typescript_sdk": "TypeScript SDK",
        }[component]
        print(f"  {component_name}: {current_versions[component]} → {info['version']}")

    # Confirm (skipped in headless mode or with --yes)
    if not headless and not args.yes:
        response = input("\nProceed with updates? (y/n): ").strip().lower()
        if response != "y":
            print("Cancelled.")
            sys.exit(0)

    # Apply all updates
    updater.update_all(updates, current_versions)

    print("\nVersion updates complete!")
    print("\nDon't forget to:")
    print("  - Review the changes with `git diff`")
    print("  - Run `uv lock` to refresh the lockfile")
    print("  - Commit the changes")
    print("  - Create git tags for the new versions")
    print("  - Push the changes and tags")


if __name__ == "__main__":
    main()

# Public Release Checklist

Before uploading to GitHub:

- Confirm the author list in `CITATION.cff`.
- Replace `https://github.com/USERNAME/REPOSITORY` with the real GitHub URL.
- Replace `10.xxxx/zenodo.xxxxxxx` after Zenodo creates the DOI.
- Choose final licenses and replace `LICENSE_NOTES.md` with real license files if needed.
- Check that no Word drafts, reviewer comments, local agent files, private notes or temporary files are present.
- Confirm that the included tables are acceptable to share as processed data derived from public PDB records.

Suggested GitHub and Zenodo flow:

1. Create a new public GitHub repository.
2. Copy the contents of this folder into the new repository.
3. Commit and push the files.
4. In Zenodo, enable GitHub integration for the repository.
5. Create a GitHub release, for example `v1.0.0`.
6. Zenodo will archive that release and mint a DOI.
7. Update `README.md`, `DATA_AVAILABILITY.md` and `CITATION.cff` with the final GitHub URL and DOI.
8. Create a small follow-up GitHub release if the DOI placeholders were updated after the first archive.


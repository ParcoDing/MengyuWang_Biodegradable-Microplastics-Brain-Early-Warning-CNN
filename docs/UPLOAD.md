# Upload and release guide

This is a prepared local payload. No remote repository was modified during preparation.

The target repository contained only a short README on inspection. Copy the **contents** of this package into a local clone of the target repository so `README.md`, `data/`, `scripts/` and `results/` appear at the repository root. Do not upload the enclosing folder as an additional nesting level. Preserve the package's own `.gitignore` and `.gitattributes`; never copy an unrelated `.git` directory.

Review `REPRODUCIBILITY.md` and the Chinese preparation note before the final public release. A partial supporting-data release can be described accurately as such; it should not claim complete CNN training reproducibility.

Example commands after copying into your clone:

```bash
python scripts/validate_package.py
git status --short
git add README.md CITATION.cff LICENSE_STATUS.md .gitignore .gitattributes requirements*.txt data results scripts docs
git commit -m "Add brain imaging data and downstream analyses"
# Review the commit, then push to your intended branch.
```

Use Git/GitHub Desktop to upload the directory tree. GitHub documents a 25 MiB per-file browser upload limit and blocks Git files above 100 MiB; the package validation reports file sizes. The delivery ZIP is a transport archive and should be extracted, not committed as the repository's sole content. See [GitHub large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

After author review, consider a tagged release and a durable repository archive with a DOI. Do not put a made-up DOI, publication status, or blanket data-availability claim into the manuscript. Consult the current [PNAS editorial policies](https://www.pnas.org/author-center/editorial-and-journal-policies) for the final submission; this preparation is not a certification of journal compliance.

Suggested availability wording for the **current scope**, only after the files are actually public:

> Processed brain images, segmentation masks, predefined image features, stored CNN scores, and downstream analysis scripts are available at https://github.com/ParcoDing/MengyuWang_Biodegradable-Microplastics-Brain-Early-Warning-CNN. The repository documentation describes the available materials and current reproducibility limitations.

Expand this statement to cover the complete study only after the missing data and training materials have been deposited.

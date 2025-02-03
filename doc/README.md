# Documentation Setup Guide

This guide provides a step-by-step process to set up the documentation for the project.

## Go into Docs Folder

```bash
cd doc
```

## Optional: Clean/Remove Existing Files

```bash
make clean
```

## Build the Documentation

```bash
make html
```

## Serve the Documentation Locally to Check Changes

```bash
cd _build/html
poetry run python3 -m http.server
```

This will serve the documentation on `http://localhost:8000/`.


## Update ReadTheDocs

todo: still needs to be set up but will probably work by pushing any changes to the .rst files and updating master/main.

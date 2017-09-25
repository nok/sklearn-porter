# Changelog

All notable changes to this project will be documented in this file.


## [Unreleased]

### Added

- Add [changelog.md](changelog.md) for the next releases.
- Add [changelog.md](changelog.md) and [readme.md](readme.md) to each build of a release.
- Add for each target programming language a new new command-line argument (e.g. `--java`, `--c` or `--go`).
- Add pipe functionality and the related command-line argument (`--pipe` or `-p`).
- Add test class `Go` in `tests/language/Go.py` to test all implementations for the target programming language Go. 
 
### Changed
 
- Use human-readable placeholders (e.g. `'{class_name}.{method_name}'`) instead of index-based placeholders (e.g. `'{0}.{1}'`) in all main templates of all estimators.
 
### Removed

- Hide the command-line argument `--language` and `-l` for the choice of the target programming language. 
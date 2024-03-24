# Pixaxe

Pixaxe is an indexed pixel editor. It aims to be a scriptable, easy-to-use
palette-based pixel art editor.

Right now, it's an experiment.

This project will be lightly commercialized by only providing official builds
through a platform such as itch.io and maybe optional social services for
publishing and sharing pixel art.

## Goals

- Image edits are transactional and history is preserved automatically.

  Because everything is indexed, edits can be stored as a "wrapping-add" for the
  color index to go from the current value to the new value. Edits can be undone
  by performing a "wrapping-sub" instead.
- Muse scripting is used to:
  - Write custom commands
  - Write custom brushes
  - Write custom layer blending functions
  - Write dynamic layers
  - Define shortcut mappings

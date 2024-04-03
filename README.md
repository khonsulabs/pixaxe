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

## Licenses

The [`pixaxe/`](./pixaxe/) directory is licensed under the [Pixaxe
License](./pixaxe/LICENSE.md), while all other contents of this repository are
licensed under the [MIT License](./pixaxe-core/LICENSE). The Pixaxe License is a
restricted, source-available license that has a future license grant of the MIT
License after two years.

Redistributing unmodified copies of Pixaxe and/or its source made available
under these licenses will always be permitted.

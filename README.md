# Pxli

*Pxli* is an indexed pixel editor. It aims to be a scriptable, easy-to-use
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

## Copyright and License

*Pxli*, the indexed pixel editor
Copyright (C) 2024 Khonsu Labs

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

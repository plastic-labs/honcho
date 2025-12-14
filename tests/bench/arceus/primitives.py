"""Primitive transformation operations for ARC-AGI-2 tasks."""

from typing import Any, Callable, List, Optional

import numpy as np

Grid = List[List[int]]


class ARCPrimitives:
    """Collection of primitive transformation operations for ARC tasks."""

    @staticmethod
    def rotate_90(grid: Grid) -> Grid:
        """Rotate grid 90 degrees clockwise."""
        arr = np.array(grid)
        rotated = np.rot90(arr, k=-1)
        return rotated.tolist()

    @staticmethod
    def rotate_180(grid: Grid) -> Grid:
        """Rotate grid 180 degrees."""
        arr = np.array(grid)
        rotated = np.rot90(arr, k=2)
        return rotated.tolist()

    @staticmethod
    def rotate_270(grid: Grid) -> Grid:
        """Rotate grid 270 degrees clockwise (90 counter-clockwise)."""
        arr = np.array(grid)
        rotated = np.rot90(arr, k=1)
        return rotated.tolist()

    @staticmethod
    def flip_horizontal(grid: Grid) -> Grid:
        """Flip grid horizontally (left-right)."""
        arr = np.array(grid)
        flipped = np.fliplr(arr)
        return flipped.tolist()

    @staticmethod
    def flip_vertical(grid: Grid) -> Grid:
        """Flip grid vertically (up-down)."""
        arr = np.array(grid)
        flipped = np.flipud(arr)
        return flipped.tolist()

    @staticmethod
    def tile_grid(grid: Grid, rows: int = 3, cols: int = 3) -> Grid:
        """Tile the grid in a rows x cols pattern."""
        arr = np.array(grid)
        tiled = np.tile(arr, (rows, cols))
        return tiled.tolist()

    @staticmethod
    def extract_objects(grid: Grid) -> List[tuple[Grid, int, int]]:
        """
        Extract connected components (objects) from grid.

        Returns:
            List of (object_grid, row_offset, col_offset)
        """
        arr = np.array(grid)
        objects = []
        visited = np.zeros_like(arr, dtype=bool)

        def flood_fill(r: int, c: int, color: int) -> List[tuple[int, int]]:
            if (
                r < 0
                or r >= arr.shape[0]
                or c < 0
                or c >= arr.shape[1]
                or visited[r, c]
                or arr[r, c] != color
                or color == 0
            ):
                return []

            visited[r, c] = True
            cells = [(r, c)]
            cells.extend(flood_fill(r + 1, c, color))
            cells.extend(flood_fill(r - 1, c, color))
            cells.extend(flood_fill(r, c + 1, color))
            cells.extend(flood_fill(r, c - 1, color))
            return cells

        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if not visited[r, c] and arr[r, c] != 0:
                    cells = flood_fill(r, c, arr[r, c])
                    if cells:
                        rows, cols = zip(*cells)
                        min_r, max_r = min(rows), max(rows)
                        min_c, max_c = min(cols), max(cols)

                        obj_grid = [
                            [0 for _ in range(max_c - min_c + 1)]
                            for _ in range(max_r - min_r + 1)
                        ]
                        for r, c in cells:
                            obj_grid[r - min_r][c - min_c] = arr[r, c]

                        objects.append((obj_grid, min_r, min_c))

        return objects

    @staticmethod
    def count_colors(grid: Grid) -> dict[int, int]:
        """Count occurrences of each color in the grid."""
        arr = np.array(grid)
        unique, counts = np.unique(arr, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    @staticmethod
    def replace_color(grid: Grid, old_color: int, new_color: int) -> Grid:
        """Replace all instances of old_color with new_color."""
        arr = np.array(grid)
        arr[arr == old_color] = new_color
        return arr.tolist()

    @staticmethod
    def get_symmetry(grid: Grid) -> dict[str, bool]:
        """Check for various symmetries in the grid."""
        arr = np.array(grid)
        return {
            "horizontal": np.array_equal(arr, np.fliplr(arr)),
            "vertical": np.array_equal(arr, np.flipud(arr)),
            "rotational_90": np.array_equal(arr, np.rot90(arr, k=1)),
            "rotational_180": np.array_equal(arr, np.rot90(arr, k=2)),
        }

    @staticmethod
    def scale_grid(grid: Grid, factor: int) -> Grid:
        """Scale grid by repeating each cell factor x factor times."""
        arr = np.array(grid)
        scaled = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
        return scaled.tolist()

    @staticmethod
    def crop_grid(
        grid: Grid, start_row: int, start_col: int, height: int, width: int
    ) -> Grid:
        """Crop a rectangular region from the grid."""
        arr = np.array(grid)
        cropped = arr[start_row : start_row + height, start_col : start_col + width]
        return cropped.tolist()

    @staticmethod
    def pad_grid(
        grid: Grid, pad_rows: int, pad_cols: int, fill_value: int = 0
    ) -> Grid:
        """Pad grid with fill_value."""
        arr = np.array(grid)
        padded = np.pad(
            arr, ((pad_rows, pad_rows), (pad_cols, pad_cols)), constant_values=fill_value
        )
        return padded.tolist()

    @staticmethod
    def detect_pattern(grid: Grid) -> dict[str, Any]:
        """Analyze grid to detect patterns and properties."""
        arr = np.array(grid)
        colors = ARCPrimitives.count_colors(grid)
        symmetries = ARCPrimitives.get_symmetry(grid)

        return {
            "shape": arr.shape,
            "num_colors": len(colors),
            "colors": colors,
            "symmetries": symmetries,
            "is_square": arr.shape[0] == arr.shape[1],
            "num_objects": len(ARCPrimitives.extract_objects(grid)),
        }

    @staticmethod
    def transpose(grid: Grid) -> Grid:
        """Transpose the grid (swap rows and columns)."""
        arr = np.array(grid)
        return arr.T.tolist()

    @staticmethod
    def fill_background(grid: Grid, color: int = 0) -> Grid:
        """Fill all background (0) cells with specified color."""
        arr = np.array(grid)
        arr[arr == 0] = color
        return arr.tolist()

    @staticmethod
    def invert_colors(grid: Grid, max_color: int = 9) -> Grid:
        """Invert all colors (0->9, 1->8, etc)."""
        arr = np.array(grid)
        return (max_color - arr).tolist()

    @staticmethod
    def extract_largest_object(grid: Grid) -> Grid:
        """Extract the largest connected object from grid."""
        objects = ARCPrimitives.extract_objects(grid)
        if not objects:
            return grid

        # Find largest object by area
        largest = max(objects, key=lambda obj: len(obj[0]) * len(obj[0][0]))
        return largest[0]

    @staticmethod
    def compress_grid(grid: Grid) -> Grid:
        """Remove all-zero rows and columns."""
        arr = np.array(grid)

        # Find non-zero rows and columns
        non_zero_rows = np.any(arr != 0, axis=1)
        non_zero_cols = np.any(arr != 0, axis=0)

        # Compress
        if not np.any(non_zero_rows) or not np.any(non_zero_cols):
            return [[0]]  # Empty grid

        compressed = arr[non_zero_rows][:, non_zero_cols]
        return compressed.tolist()

    @staticmethod
    def overlay_grids(grid1: Grid, grid2: Grid, mode: str = "or") -> Grid:
        """
        Overlay two grids.

        Modes:
        - 'or': Take non-zero value from either grid
        - 'and': Only keep cells that are non-zero in both
        - 'xor': Take cells that differ
        """
        arr1 = np.array(grid1)
        arr2 = np.array(grid2)

        # Ensure same shape
        if arr1.shape != arr2.shape:
            return grid1

        if mode == "or":
            result = np.where(arr1 != 0, arr1, arr2)
        elif mode == "and":
            result = np.where((arr1 != 0) & (arr2 != 0), arr1, 0)
        elif mode == "xor":
            result = np.where(arr1 == arr2, 0, arr1)
        else:
            result = arr1

        return result.tolist()

    @staticmethod
    def apply_to_each_object(grid: Grid, operation: str) -> Grid:
        """Apply an operation to each connected object separately."""
        objects = ARCPrimitives.extract_objects(grid)
        if not objects:
            return grid

        result = np.zeros_like(np.array(grid))

        for obj_grid, row_offset, col_offset in objects:
            # Apply operation
            if operation == "rotate_90":
                transformed = ARCPrimitives.rotate_90(obj_grid)
            elif operation == "flip_horizontal":
                transformed = ARCPrimitives.flip_horizontal(obj_grid)
            elif operation == "flip_vertical":
                transformed = ARCPrimitives.flip_vertical(obj_grid)
            else:
                transformed = obj_grid

            # Place back
            arr = np.array(transformed)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    r, c = row_offset + i, col_offset + j
                    if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                        if arr[i, j] != 0:
                            result[r, c] = arr[i, j]

        return result.tolist()

    @staticmethod
    def gravity_down(grid: Grid, color: Optional[int] = None) -> Grid:
        """Apply gravity - make objects fall down."""
        arr = np.array(grid)
        result = np.zeros_like(arr)

        for col in range(arr.shape[1]):
            # Get non-zero (or specified color) cells in this column
            if color is None:
                cells = [arr[row, col] for row in range(arr.shape[0]) if arr[row, col] != 0]
            else:
                cells = [arr[row, col] for row in range(arr.shape[0]) if arr[row, col] == color]

            # Place them at bottom
            for i, cell_val in enumerate(cells):
                result[arr.shape[0] - len(cells) + i, col] = cell_val

            # Keep other colors in place
            if color is not None:
                for row in range(arr.shape[0]):
                    if arr[row, col] != 0 and arr[row, col] != color:
                        result[row, col] = arr[row, col]

        return result.tolist()

    @staticmethod
    def repeat_pattern(grid: Grid, rows: int, cols: int) -> Grid:
        """Repeat the grid pattern (similar to tile but preserves boundaries)."""
        return ARCPrimitives.tile_grid(grid, rows, cols)

    @staticmethod
    def draw_border(grid: Grid, color: int, thickness: int = 1) -> Grid:
        """Draw a border around the grid."""
        arr = np.array(grid)
        result = arr.copy()

        # Top and bottom
        for t in range(thickness):
            if t < arr.shape[0]:
                result[t, :] = color
                result[-(t + 1), :] = color

        # Left and right
        for t in range(thickness):
            if t < arr.shape[1]:
                result[:, t] = color
                result[:, -(t + 1)] = color

        return result.tolist()


# Dictionary mapping primitive names to functions
PRIMITIVE_FUNCTIONS: dict[str, Callable] = {
    "rotate_90": ARCPrimitives.rotate_90,
    "rotate_180": ARCPrimitives.rotate_180,
    "rotate_270": ARCPrimitives.rotate_270,
    "flip_horizontal": ARCPrimitives.flip_horizontal,
    "flip_vertical": ARCPrimitives.flip_vertical,
    "tile_grid": ARCPrimitives.tile_grid,
    "extract_objects": ARCPrimitives.extract_objects,
    "count_colors": ARCPrimitives.count_colors,
    "replace_color": ARCPrimitives.replace_color,
    "get_symmetry": ARCPrimitives.get_symmetry,
    "scale_grid": ARCPrimitives.scale_grid,
    "crop_grid": ARCPrimitives.crop_grid,
    "pad_grid": ARCPrimitives.pad_grid,
    "detect_pattern": ARCPrimitives.detect_pattern,
    "transpose": ARCPrimitives.transpose,
    "fill_background": ARCPrimitives.fill_background,
    "invert_colors": ARCPrimitives.invert_colors,
    "extract_largest_object": ARCPrimitives.extract_largest_object,
    "compress_grid": ARCPrimitives.compress_grid,
    "overlay_grids": ARCPrimitives.overlay_grids,
    "apply_to_each_object": ARCPrimitives.apply_to_each_object,
    "gravity_down": ARCPrimitives.gravity_down,
    "repeat_pattern": ARCPrimitives.repeat_pattern,
    "draw_border": ARCPrimitives.draw_border,
}

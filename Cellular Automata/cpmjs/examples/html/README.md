## Notes about ObstaclesGrid and ObstaclesSimple

### Cells

It is said in the assignment: "start from the cells that kept moving even at large densities". These are keratocyte-like cells. I used the classical parameters from "Crawling and Gliding: A Computational Model for Shape-Driven Cell Migration"/ the self-study  part:
- Volume: 500
- LAMBDA_V: 50
- Perimeter: 340
- LAMBDA_P: 2
- MAX_ACT: 80
- LAMBDA_ACT: 300 (with lower values cells move slowly)

### Obstacles:
- LAMBDA_V: 200
- LAMBDA_P: 100 - these two settings keep an obstacle still
- In ObstaclesGrid Volume = 250, because in the Assignment point 3 it's written that an obstacle shoud be a half size of a cell (volume = 500)
- Perimeter: 150. This gave the most circular form with Volume = 250

### Adhesion:
Specified based on the following logic: obstacles are still objects and do not have any adhesion, so adhesion_obstacle_obstacle should be 0. Adhesion between all objects and background is "standard" (used in example modes) and equals 20. Adhesion between cells should be higher, so adhesion_cell_cell = 40. 

Not sure about the logic for adhesion_obstacle_cell. If it is set to 0 when a cell encounters an obstacle it gets stuck, so I set it to 10 for now. 20 works similarly. Value to play around with.

### Grid:
I implemented the initial arrangement of evenly spaced obstacles and cells in `ObstaclesGrid`  `initializeGrid`. 

The number of obstacles can be easily configured with `OBSTACLES_PER_ROW` and `OBSTACLES_PER_COLUMN`, the number of cells: `MOTILE_CELLS_NUMBER`

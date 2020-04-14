let CPM = require("../../build/cpm-cjs.js")

// Simulation code here.
let config = {

	ndim : 2,
	field_size : [250,250],
	conf : {
		T : 20,                             // CPM temperature

		// Adhesion parameters
		J : [ [0,20,20],
			[20,0,20],                    // Cells
			[20,20,0]],                // Obstacles

		// VolumeConstraint parameters
		LAMBDA_V : [0,50,200],				// VolumeConstraint importance per cellkind
		V : [0,500,250],                   // Target volume of each cellkind

		// PerimeterConstraint parameters
		LAMBDA_P : [0,2,100],				    // PerimeterConstraint importance per cellkind
		P : [0,340,150],					    // Target perimeter of each cellkind

		// ActivityConstraint parameters
		LAMBDA_ACT : [0,200,0],			    // ActivityConstraint importance per cellkind
		MAX_ACT : [0,80,0],				    // Activity memory duration per cellkind
		ACT_MEAN : "geometric",				// Is neighborhood activity computed as a
																				// "geometric" or "arithmetic" mean?
	},
	simsettings : {

		// Cells on the grid
		NRCELLS : [0, 0],					// Number of cells to seed for all
																				// non-background cellkinds.
		// Runtime etc
		BURNIN : 500,
		RUNTIME : 8000,

		// Visualization
		CANVASCOLOR : "eaecef",
		CELLCOLOR : ["000000","0245ab"],
		ACTCOLOR : [true,false],			// Should pixel activity values be displayed?
		SHOWBORDERS : [true, true],			// Should cellborders be displayed?
		BORDERCOL : ["DDDDDD","DDDDDD"],

		zoom : 2,							// zoom in on canvas with this factor.

		// Output images
		SAVEIMG : true,

		IMGFRAMERATE : 1,
		SAVEPATH : "output/img/ObstaclesGrid",	// ... And save the image in this folder.
		EXPNAME : "Obstacles",

		LOGRATE : 10							// zoom in on canvas with this factor.
	}
}

let MOTILE_CELLS_NUMBER = 100
let OBSTACLES_PER_ROW = 2
let OBSTACLES_PER_COLUMN = 2

// java script does not support testing of array for equality
// and neither it supports operators overloading
// so a little hack
function arrayHasItem(array, item) {
	let array_str = JSON.stringify(array);
	let item_str = JSON.stringify(item);
	let index = array_str.indexOf(item_str);
	return index != -1;
}

let custommethods = {
	initializeGrid : initializeGrid
}

let sim = new CPM.Simulation( config, custommethods )
sim.run()

function initializeGrid(){

	// add the initializer if not already there
	if( !this.helpClasses["gm"] ){ this.addGridManipulator() }

	var used_coordinates = []
	// Seed obstacles
	if (OBSTACLES_PER_ROW > 0 && OBSTACLES_PER_COLUMN > 0) {
		let step_x = Math.ceil(this.C.extents[0]/OBSTACLES_PER_ROW)
		let step_y = Math.ceil(this.C.extents[0]/OBSTACLES_PER_COLUMN)
		let x_init = Math.floor(step_x/2)
		let y_init = Math.floor(step_y/2)
		for( var x = x_init ; x < this.C.extents[0] ; x += step_x ){
			for( var y = y_init ; y < this.C.extents[1] ; y += step_y ){
				this.gm.seedCellAt( 2, [x,y] )
				// if a new cell is seeded with the same coordinates as one of the previous cells
				// it overrides the cell with matching coordinates
				// to prevent this we keep all the used coordinates
				// and check that each new cell is added with unique coordinates
				used_coordinates.push([x,y])
			}
		}
	}


	// Seed motile cells
	for (var i=0; i < MOTILE_CELLS_NUMBER; i+=1) {
		var x = Math.floor(Math.random() * this.C.extents[0])
		var y = Math.floor(Math.random() * this.C.extents[1])
		while (arrayHasItem(used_coordinates, [x,y])) {
			x = Math.floor(Math.random() * this.C.extents[0])
			y = Math.floor(Math.random() * this.C.extents[1])
		}

		this.gm.seedCellAt( 1, [x, y])
		used_coordinates.push([x,y])
	}
}

let CPM = require("../../build/cpm-cjs.js")

// Simulation code here.
let config = {

	ndim : 2,
	field_size : [300,300],
	conf : {
		T : 20,                             // CPM temperature

		// Adhesion parameters
		J : [ [0,20,20],
			[20,40,20],                    // Cells
			[20,20,0]],                // Obstacles

		// VolumeConstraint parameters
		LAMBDA_V : [0,50,200],				// VolumeConstraint importance per cellkind
		V : [0,500,600],                   // Target volume of each cellkind

		// PerimeterConstraint parameters
		LAMBDA_P : [0,2,100],				    // PerimeterConstraint importance per cellkind
		P : [0,340,250],					    // Target perimeter of each cellkind

		// ActivityConstraint parameters
		LAMBDA_ACT : [0,300,0],			    // ActivityConstraint importance per cellkind
		MAX_ACT : [0,80,0],				    // Activity memory duration per cellkind
		ACT_MEAN : "geometric",				// Is neighborhood activity computed as a
																				// "geometric" or "arithmetic" mean?
	},
	simsettings : {

		// Cells on the grid
		NRCELLS : [1,10],					// Number of cells to seed for all
																				// non-background cellkinds.
		// Runtime etc
		BURNIN : 500,
		RUNTIME : 1000,
		RUNTIME_BROWSER : "Inf",

		// Visualization
		CANVASCOLOR : "eaecef",
		CELLCOLOR : ["990000","000000"],
		ACTCOLOR : [false,false],			// Should pixel activity values be displayed?
		SHOWBORDERS : [true, true],			// Should cellborders be displayed?
		BORDERCOL : ["DDDDDD","DDDDDD"],
		zoom : 2,							// zoom in on canvas with this factor.

		// Output images
		SAVEIMG : true,

		IMGFRAMERATE : 1,
		SAVEPATH : "output/img/ObstaclesSimple",	// ... And save the image in this folder.
		EXPNAME : "Obstacles",

		LOGRATE : 10
	}
}

let sim = new CPM.Simulation( config )
sim.run()

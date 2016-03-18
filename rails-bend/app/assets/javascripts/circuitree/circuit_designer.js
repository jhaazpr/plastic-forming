var sr_model;
var artboard;
CircuitDesigner.BLANK_CANVAS = 1;
CircuitDesigner.UNIT_ADD = 2;
CircuitDesigner.ARTBOARD_ADD = 3;

function CircuitDesigner(container){
	sr_model = new SheetResistanceModel(10);
	this.paper = paper;
	this.container = container;
	this.temp = null;
	this.init();
	this.makeLayers();

	this.animations = [];
	this.update();
	var self = this;
	this.animation_handler = new AnimationHandler(paper);
	this.state = {};
}

CircuitDesigner.prototype = {
	makeLayers: function(){
		while(paper.project.layers.length > 0) paper.project.layers[0].remove();
		
		this.layer = new paper.Layer({ name: "EL: Ellustrator SVG" });
		this.art_layer = new ArtworkLayer(paper, this.layer);
		this.circuit_layer = new CircuitLayer(paper, this.layer, "SI");
		this.paper.view.update();
	},
	init: function(){
		// setups paperjs 
		var c = this.container;
		this.canvas = DOM.tag("canvas")
				.prop('resize', true)
				.height(c.height() - 50)
				.width(c.width());

		c.append(this.canvas);	

		this.paper = new paper.PaperScope();
		this.paper.setup(this.canvas[0]);
		this.height = this.paper.view.size.height;
		this.width = this.paper.view.size.width;

	    zoom = new Zoom(this.paper.view.zoom, this.paper);

		this.paper.view.zoom = 1;	
		var scope = this; 
		
		// Setups tools
	    this.toolbox = new Toolbox(this.paper, $("#toolbox"));	
		return this;
	},
	update: function(){
		if(typeof this.paper == "undefined") return;
		paper.view.update();
	},

	clear: function(){
		this.makeLayers();
		this.update();
	},
	addSVG: function(filename, position, callback){
		var scope = this;
		var fileType = filename.split('/');
		fileType = fileType[fileType.length - 1];
		fileType = fileType.split('_');
		fileType = fileType[0].toLowerCase();
		console.log("addSVG: filename", fileType, filename);
		this.paper.project.importSVG(filename, {
	    	onLoad: function(item) { 
		    	item.position = position;
				eSVG = new BenderSVG(item, scope, CircuitDesigner.UNIT_ADD);
	    		hm.save();
	    		zoom.home();
	    		zoom.home();
	    	}
		});
	},
	loadJSON: function(json, clearCanvas){
		var scope = this;
		if(clearCanvas == CircuitDesigner.BLANK_CANVAS){
			this.clear();
			sys.log("Clearing canvas!");
		}
		var item = this.paper.project.importJSON(json); 
		item[0].remove();

		eSVG = new BenderSVG(item[0], scope, CircuitDesigner.ARTBOARD_ADD);
   		scope.update();
   		$('#shade').fadeOut(1000);
	},
	json: function(){
		return paper.project.exportJSON({
			asString: true,
			precision: 5
		});
	}, 
	svg: function(){
		this.clearForSave();
		var filename = $('#design-name b').html().trim().replace(/ /g, "_").toLowerCase();
 

		if(_.isUndefined(filename)) filename = "export";
		if(filename == "") filename = "export"; 
		filename = filename.split('.')[0];

		
		console.log("Exporting file as SVG");
		zoom = 1;

		paper.view.update();

		exp = paper.project.exportSVG({ 
			asString: true,
			precision: 5
		});

		saveAs(new Blob([exp], {type:"application/svg+xml"}), filename + ".svg")
		this.unclearForSave();
	},	
	clearForSave: function(){
		this.state.tool = this.toolbox.clearTool();
		this.temp = getTerminalHelpers();
		this.temp = _.map(this.temp, function(el, i, arr){
			var e = {parent: el.parent, el: el}
			el.remove();
			return e;
		});
		this.circuit_layer.legend.remove();		
	}, 
	unclearForSave: function(){
		if(!_.isNull(this.state.tool)){
			console.log("Tool reenable", this.state.tool.name);
			this.state.tool.dom.addClass('btn-warning').removeClass('btn-Bender');
			this.toolbox.reenable(this.state.tool.name);
		}
		if(!_.isNull(this.temp)){
			_.each(this.temp, function(el, i, arr){
				el.parent.addChild(el.el);
			});
			this.temp = null;
		}

		this.paper.view.update();
	}
}
var getTerminalHelpers = function(){
	return BenderSVG.get({terminal_helper: true});
}


                 

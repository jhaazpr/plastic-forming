function CanvasPanTool(paper){
	this.paper = paper;
	this.tool = new paper.Tool();
	var scope = this;


	this.tool.onMouseDown = function(event){
		
	}

	this.tool.onMouseUp = function(event){
	
	}

	this.tool.onMouseDrag = function(event){
		_.each(paper.project.layers, function(el, i, arr){
			el.position.x += event.delta.x;
			el.position.y += event.delta.y;
			scope.update();
		});
	}
	this.tool.onPinchMove = function(event){
		sys.show("Zooming to:" + event.scale);
		paper.view.zoom = event.scale;
		paper.view.update();
	}		
}


CanvasPanTool.prototype = {
	enable: function(){
		sys.log("Activated the Canvas Pan Tool.");
		var scope = this;
	   	this.hammertime = Hammer($('canvas')[0]);
	   	this.hammertime.get('pinch').set({ enable: true });
	    this.hammertime.on('tap', scope.tool.onTap);
	    this.hammertime.on('pinchstart', scope.tool.onPinchStart);
	    this.hammertime.on('pinchend', scope.tool.onPinchEnd);
	    this.hammertime.on('pinchmove', scope.tool.onPinchMove);
	},
	disable: function(){
	   console.log("Canvas Pan Disabled");
	   var scope  = this;
	   this.hammertime.off('tap', scope.tool.onTap);
	   this.hammertime.off('pinchstart', scope.tool.onPinchStart);
	   this.hammertime.off('pinchend', scope.tool.onPinchEnd);
	   this.hammertime.off('pinchmove', scope.tool.onPinchMove);
	},
	update: function(){
		this.paper.view.update();
	}, 
	clear: function(){
		
	},
	selectAll: function(flag){
		this.paper.project.activeLayer.selected = flag;
	}, 
	setSVG: function(svg){
		this.svg = svg;
	}
}
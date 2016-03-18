
var hitOptions = {
	segments: true,
	stroke: true,
	fill: true,
	tolerance: 15
};

var HIT_TEST_BOUNDING_RADIUS = 15;


function DeleteTool(paper){
	this.paper = paper;
	this.name = "DeleteTool";

	this.tool = new paper.Tool();
	this.tool.distanceThreshold = 10;
	this.sm = new SelectionManager(paper);

	var scope = this;

	var whatDidIHit = function(positionOnCanvas){
		hitResult = scope.paper.project.hitTest(positionOnCanvas, hitOptions);	
		if(_.isNull(hitResult)) return {type: "canvas", result: hitResult}
		else if(hitResult.item.name == "NCB: artboard") return {type: "canvas", result: hitResult}
		else{ return {type: "element", result: hitResult} }
	}
	var route = function(event, allowableSubEvents, eventFN){
		// Recalibrate event point geometry

		if(_.isUndefined(event.center)){
			var positionOnCanvas = event.point;
		}else{
			event.center.y -= $('canvas').offset().top;
			var positionOnCanvas = scope.paper.view.viewToProject(new paper.Point(event.center.x, event.center.y));
		}
		// START
		if(["mousedown", "mousedrag"].indexOf(event.type) != -1){
			var hit = whatDidIHit(positionOnCanvas);
			var hitType = hit.type;
			var hitResult = hit.result;
			scope.canvas_item_type = hitType;
			if(allowableSubEvents.indexOf(scope.canvas_item_type) != -1){
				scope[scope.canvas_item_type][eventFN](event, hitResult, scope);
			}
		}

	}
	
	this.tool.onMouseDown = function(event){
		route(event, ["element"], "onMouseDown");
	}
	this.tool.onMouseDrag = function(event){
		route(event, ["element"], "onMouseDrag")
	}
}


DeleteTool.prototype = {
	enable: function(){
	   var scope = this;
	   sys.log("Activated the delete tool.");
	   // this.hammertime = Hammer($('canvas')[0]);
	   // this.hammertime.on('tap', scope.tool.onTap);
	},
	disable: function(){
	   console.log("Touch Tools Disabled");
	   var scope  = this;
	   // this.hammertime.off('tap', scope.tool.onTap);
	   this.selectAll(false);
	   this.update();
	},
	update: function(){
		this.paper.view.update();
	}, 
	clear: function(){
	},
	selectAll: function(flag){
    	this.paper.project.activeLayer.selected = flag;
    	designer.circuit_layer.layer.selected = flag;
 	}, 
	element: {
		onMouseDrag: function(event, hitResult, scope){
			var cluster = hitResult.item;
			while(_.isUndefined(cluster.canvasItem))
				cluster = cluster.parent;
			scope.sm.add(cluster, event.event.shiftKey);
			scope.sm.remove();
			hm.save();
		},
		onMouseDown: function(event, hitResult, scope){
			console.log("onMouseDown");

			var cluster = hitResult.item;
			while(_.isUndefined(cluster.canvasItem))
				cluster = cluster.parent;
			scope.sm.add(cluster, event.event.shiftKey);
			scope.sm.remove();
			hm.save();
		}
	},
	transform: {
	},
	canvas: {
	},
	pan:  {
	}
}




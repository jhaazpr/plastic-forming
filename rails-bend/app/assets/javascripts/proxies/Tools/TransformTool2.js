
var hitOptions = {
	segments: true,
	stroke: true,
	fill: true,
	tolerance: 15
};
TransformTool2.ROTATING = 1;
TransformTool2.SCALING = 2;
TransformTool2.TOUCH = 0;
var HIT_TEST_BOUNDING_RADIUS = 15;

function tS(v){
	if(v == TransformTool2.ROTATING)
		return "TRANSFORM.ROTATE";
	if(v == TransformTool2.SCALING)
		return "TRANSFORM.SCALE";
	return "TRANSFORM.NULL";
}

function TransformTool2(paper){
	this.paper = paper;
	this.name = "TransformTool2";

	this.tool = new paper.Tool();
	this.tool.distanceThreshold = 10;
	this.sm = new SelectionManager(paper);
	this.pinching = false;
	this.rotating = false;
	this.dragged = false;
	var scope = this;

	var whatDidIHit = function(positionOnCanvas){
		hitResult = scope.paper.project.hitTest(positionOnCanvas, hitOptions);	
		console.log(hitResult.item.name);
		if(_.isNull(hitResult)) return {type: "canvas", result: hitResult}
		else if(hitResult.item.name == "NCB: artboard") return {type: "canvas", result: hitResult}
		else{ return {type: "element", result: hitResult} }
	}
	var route = function(event, allowableSubEvents, eventFN){
		// Recalibrate event point geometry
		// console.log("ROUTING", event);

		if(_.isUndefined(event.center)){
			var positionOnCanvas = event.point;
		}else{
			event.center.y -= $('canvas').offset().top;
			var positionOnCanvas = scope.paper.view.viewToProject(new paper.Point(event.center.x, event.center.y));
		}
		// START
		if(["mousedown", "rotatestart", "pinchstart"].indexOf(event.type) != -1){
			var hit = whatDidIHit(positionOnCanvas);
			// console.log("HIT", hit);
			var hitType = hit.type;
			var hitResult = hit.result;
			scope.canvas_item_type = hitType;
			// console.log("Hit test", scope.canvas_item_type, eventFN);		
			if(allowableSubEvents.indexOf(scope.canvas_item_type) != -1){
				scope[scope.canvas_item_type][eventFN](event, hitResult, scope);
			}
		}
		// DRAG	
		else if(["rotatemove", "mousedrag", "pinchmove"].indexOf(event.type) != -1) {
			// Route accordingly...		
			var hit = whatDidIHit(positionOnCanvas);
			// console.log("HIT", hit);
			var hitType = hit.type;
			var hitResult = hit.result;
			scope.canvas_item_type = hitType;

			// console.log("Hit test", scope.canvas_item_type, eventFN);	
			if(allowableSubEvents.indexOf(scope.canvas_item_type) != -1){
				scope[scope.canvas_item_type][eventFN](event, scope);
			}
			scope.update();
		}
		// STOP
		else if(["rotateend","pinchend", "mouseup"].indexOf(event.type) != -1) {
			if(allowableSubEvents.indexOf(scope.canvas_item_type) != -1){
				scope[scope.canvas_item_type][eventFN](event, scope);
			}
			// hm.save();
			scope.update();
			// console.log("Setting to null")
			// scope.canvas_item_type = null;
		}
	}
	// this.tool.onTap = function(event){
	// 	route(event, ["canvas", "element"], "onTap");
	// }
	this.tool.onPinchMove = function(event){
		scope.pinching = true;
		route(event, ["canvas", "element"], "onPinchMove");
	}
	this.tool.onPinchStart = function(event){
		route(event, ["canvas", "element"], "onPinchStart");
	}
	this.tool.onPinchEnd = function(event){
		scope.pinching = false;
		route(event, ["canvas", "element"], "onPinchEnd");
	}
	this.tool.onRotateMove = function(event){
		scope.rotating = true;
		route(event, ["canvas", "element"], "onRotateMove");
	}
	this.tool.onRotateStart = function(event){
		route(event, ["canvas", "element"], "onRotateStart");
	}
	this.tool.onRotateEnd = function(event){
		scope.rotating = false;
		route(event, ["canvas", "element"], "onRotateEnd");
	}
	this.tool.onMouseDrag = function(event){
		route(event, ["element", "canvas"], "onMouseDrag");
	}
	this.tool.onMouseDown = function(event){
		route(event, ["element", "canvas"], "onMouseDown");
	}
	this.tool.onMouseUp = function(event){
		route(event, ["element", "canvas"], "onMouseUp");
	}
	

}


TransformTool2.prototype = {
	enable: function(){
	   var scope = this;
	   sys.log("Activated the Touch Manipulation Tool.<br> You can move and transform elements by dragging or pinching.");
	   this.hammertime = Hammer($('canvas')[0]);
	   this.hammertime.get('rotate').set({ enable: true });
	   this.hammertime.get('pinch').set({ enable: true });
	   // this.hammertime.on('tap', scope.tool.onTap);
	   this.hammertime.on('pinchstart', scope.tool.onPinchStart);
	   this.hammertime.on('pinchend', scope.tool.onPinchEnd);
	   this.hammertime.on('pinchmove', scope.tool.onPinchMove);
	   this.hammertime.on('rotatestart', scope.tool.onRotateStart);
	   this.hammertime.on('rotateend', scope.tool.onRotateEnd);
	   this.hammertime.on('rotatemove', scope.tool.onRotateMove);
	},
	reenable: function(){
	   var scope = this;
	   this.hammertime = Hammer($('canvas')[0]);
	   this.hammertime.get('rotate').set({ enable: true });
	   this.hammertime.get('pinch').set({ enable: true });
	   // this.hammertime.on('tap', scope.tool.onTap);
	   this.hammertime.on('pinchstart', scope.tool.onPinchStart);
	   this.hammertime.on('pinchend', scope.tool.onPinchEnd);
	   this.hammertime.on('pinchmove', scope.tool.onPinchMove);
	   this.hammertime.on('rotatestart', scope.tool.onRotateStart);
	   this.hammertime.on('rotateend', scope.tool.onRotateEnd);
	   this.hammertime.on('rotatemove', scope.tool.onRotateMove);
	},
	disable: function(){
	   console.log("Touch Tools Disabled");
	   var scope  = this;
	   // this.hammertime.off('tap', scope.tool.onTap);
	   this.hammertime.off('pinchstart', scope.tool.onPinchStart);
	   this.hammertime.off('pinchend', scope.tool.onPinchEnd);
	   this.hammertime.off('pinchmove', scope.tool.onPinchMove);
	   this.hammertime.off('rotatestart', scope.tool.onRotateStart);
	   this.hammertime.off('rotateend', scope.tool.onRotateEnd);
	   this.hammertime.off('rotatemove', scope.tool.onRotateMove);
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
		// onTap: function(event, hitResult, scope){
		// 	// console.log("pTap");
		// 	var cluster = hitResult.item;
		// 	while(_.isUndefined(cluster.canvasItem))
		// 		cluster = cluster.parent;
		// 	console.log("TAPPED ELEMENT");
		// 	scope.sm.add(cluster, event.srcEvent.shiftKey);
		// },
		onMouseDown: function(event, hitResult, scope){
			// console.log("mDown");
			var cluster = hitResult.item;
			while(_.isUndefined(cluster.canvasItem))
				cluster = cluster.parent;
			console.log("DOWN ELEMENT");
			scope.sm.add(cluster, event.shiftKey);
		},
		onPinchStart: function(event, scope){
			// console.log("pStart");
			scope.pinching = true;
		},
		onPinchMove: function(event, scope){
			// console.log("pMove");
			scope.pinching = true;
			scope.sm.scale(event.scale, event.scale);
		},
		onPinchEnd: function(event, scope){
			// console.log("pEnd");
			scope.pinching = false;
		},
		onRotateStart: function(event, hitResult, scope){
			console.log("rStart");
			scope.rotating = true;
		},
		onRotateMove: function(event, scope){
			console.log("rMove");
			scope.rotating = true;
			scope.sm.rotate(event.rotation);
		},
		onRotateEnd: function(event, scope){
			console.log("rEnd");
			scope.rotating = false;
		},
		onMouseDrag: function(event, scope){	
			if(scope.pinching || scope.rotating) return;

			// console.log("mousedrag", event.delta);

			scope.sm.translate(event.delta);
			scope.dragged = true;
		},
		onMouseUp: function(event, scope){
			// if(scope.dragged){
			// 	hm.save();
			// 	scope.dragged = false;
			// }
		}
	},
	canvas: {
		onPinchStart: function(event, scope){
			// console.log("pStart");
			scope.pinching = true;
		},
		onPinchMove: function(event, scope){
			// console.log("pMove");
			scope.pinching = true;
			scope.sm.scale(event.scale, event.scale);
		},
		onPinchEnd: function(event, scope){
			// console.log("pEnd");
			scope.pinching = false;
		},
		onRotateStart: function(event, hitResult, scope){
			console.log("Canvas rStart");
			scope.rotating = true;
		},
		onRotateMove: function(event, scope){
			console.log("Canvas rMove");
			scope.rotating = true;
			scope.sm.rotate(event.rotation);
		},
		onRotateEnd: function(event, scope){
			console.log("Canvas rEnd");
			scope.rotating = false;
		},
		onMouseDown: function(event, hitResult, scope){
			console.log("c_mDown");
			scope.canvas_item_type = null;

			var pos = new paper.Point(event.point.x, event.point.y);
			var c  = paper.Path.Circle({
								radius: HIT_TEST_BOUNDING_RADIUS, 
								fillColor: "red", 
								position: pos
							});

			var paths = BenderSVG.match(paper.project, {className: "Path"})
			var intersections = TracePathTool.getAllIntersections(c, paths);
	
			c.remove();
			paths = [];
			if(intersections.length > 0){
				_.each(intersections, function(el, i, arr){
					var cluster = el._curve2.path;
					while(_.isUndefined(cluster.canvasItem))
						cluster = cluster.parent;
					paths.push(cluster.id);
				});
				paths = _.uniq(paths);
				// console.log(paths[0])
				if(paths.length > 0)
					scope.sm.add(Node.get(paths[0]), false);
				else
					scope.sm.clear();
			}
			else{
				scope.sm.clear();
			}
			// 
		},
		onMouseDrag: function(event, scope){
			console.log("mousedrag canvas", event.delta);
			if(scope.pinching || scope.rotating) return;

			

			scope.sm.translate(event.delta);
			scope.dragged = true;
		},
		onMouseUp: function(event, scope){
			// if(scope.dragged){
			// 	hm.save();
			// 	scope.dragged = false;
			// }
		}
	},
	pan:  {
		onMouseDown: function(event, hitResult, scope){
			// console.log("PANNING", hitResult.type);
			// GET SELECTED ITEM
			if(["stroke", "fill", "segment"].indexOf(hitResult.type) != -1){
				var cluster = hitResult.item;
				while(_.isUndefined(cluster.canvasItem))
					cluster = cluster.parent;
				scope.sm.add(cluster, event.event.shiftKey);
			}
		},
		onMouseDrag: function(event, scope){
			// TRANSLATE
			scope.sm.translate(event.delta);
			scope.dragged = true;
		},
		onMouseUp: function(event, scope){
			// if(scope.dragged){
			// 	hm.save();
			// 	scope.dragged = false;
			// }
		}
	}
}


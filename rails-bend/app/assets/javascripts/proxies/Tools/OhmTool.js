var path;
var hitOptions = {
	segments: true,
	stroke: true,
	fill: true,
	tolerance: 0
};
OhmTool.SHORT_MESSAGE = "You are shorting your circuit. Avoid crossing any paths or connecting to terminals that aren't the same color (polarity).";

function OhmTool(paper){
	this.paper = paper;
	this.selectedPoint = null;
	this.selectedHandle = null;
	this.selectedStroke = null;
	this.selectedPath = null;
	this.lastTrace = null;

	this.tool = new paper.Tool();

	this.tool.distanceThreshold = 10;
	
	var scope = this;

	this.tool.onMouseDown = function(event){

		hitResult = scope.paper.project.hitTest(event.point, hitOptions);
		
		if(_.isNull(hitResult)) scope.canvas_item_type = "canvas";
		else{
			path = hitResult.item;
			if(path.name == "terminal") scope.canvas_item_type = "terminal";
			else if(path.name == "sticker_led") scope.canvas_item_type = "led";
			else if(path.name == "trace") scope.canvas_item_type = "trace";
			else scope.canvas_item_type = "canvas";
		} 
		console.log("MouseDown", scope.canvas_item_type);
		scope[scope.canvas_item_type].onMouseDown(event, hitResult, scope);
		scope.update();
    }
	this.tool.onMouseUp = function(event){
		console.log("MouseUp", scope.canvas_item_type);
		scope[scope.canvas_item_type].onMouseUp(event, scope);
		scope.canvas_item_type = null;
		scope.update();
	}

	this.tool.onMouseDrag = function(event){
		console.log("MouseDrag", scope.canvas_item_type);
		scope[scope.canvas_item_type].onMouseDrag(event, scope);
		scope.update();
	}	

	this.tool.onMouseMove = function(event){
		hitResult = scope.paper.project.hitTest(event.point, hitOptions);
		
		if(_.isNull(hitResult)) scope.canvas_item_type = "canvas";
		else{
			path = hitResult.item;
			if(path.name == "terminal") scope.canvas_item_type = "terminal";
			else if(path.name == "sticker_led") scope.canvas_item_type = "led";
			else if(path.name == "trace") scope.canvas_item_type = "trace";
			else scope.canvas_item_type = "canvas";
		} 
		console.log("MouseMove", scope.canvas_item_type);
		scope[scope.canvas_item_type].onMouseMove(event, hitResult, scope);
		scope.update();
	}	
	this.tool.onKeyDown = function(event){
		scope.onKeyDownDefault(event);

		if(event.key == "-" ||event.key == "backspace"){
			if(scope.lastTrace != null){
				scope.lastTrace.e_layer.remove(scope.lastTrace.guid);
				scope.clear();
			}
		}

		scope.copy_mode = event.key == "option";
		return true;
	}	

}

var trace;
var start_terminal = null;
var start_trace = null;
OhmTool.prototype = {
	enable: function(){
		designer.circuit_layer.draw_mode = true;
		designer.circuit_layer.update();
		$('#conductivity-help').show();
	},
	disable: function(){
		designer.circuit_layer.draw_mode = false;
		designer.circuit_layer.update();
		$('#conductivity-help').hide();
	},
	update: function(){
		this.paper.view.update();
	}, 
	selectAll: function(flag){
		this.paper.project.activeLayer.selected = flag;
	}, 
	setSVG: function(svg){
		this.svg = svg;
	},
	clear: function(){

	}, 
	terminal: {
		onMouseDown: function(event, hitResult, scope){
		}, 
		onMouseDrag: function(event){
		}, 
		onMouseUp: function(event, scope){
		},
		onMouseMove: function(event, hitResult, scope){
		}
	},
	trace: {
		onMouseDown: function(event, hitResult, scope){
			$('#conductivity-help').click();
		}, 
		onMouseDrag: function(event){
		}, 
		onMouseUp: function(event, scope){
		},
		onMouseMove: function(event, hitResult, scope){
			if(!_.isNull(scope.selectedPath) && scope.selectedPath != hitResult.item){
				OhmTool.glow(scope.selectedPath, false);
			}
			scope.selectedPath = hitResult.item;
			var ohmage = sr_model.apply(scope.selectedPath);
			var tolerance = 0.10;
			var l_ohmage = Math.round((1 - tolerance) * ohmage);
			var h_ohmage = Math.round((1 + tolerance) * ohmage);	
			sys.show(l_ohmage.toFixed(0) + " Ω - " + h_ohmage.toFixed(0) + " Ω" );
			OhmTool.glow(scope.selectedPath, true);
		}
	},
	led: {
		onMouseDown: function(event, hitResult, scope){
		}, 
		onMouseDrag: function(event){
		}, 
		onMouseUp: function(event){
		},
		onMouseMove: function(event, hitResult, scope){
		}
	}, 
	canvas: {
		onMouseDown: function(event, hitResult, scope){

		}, 
		onMouseDrag: function(event){
		}, 
		onMouseUp: function(event){
			terminals = designer.circuit_layer.getAllTerminals();
			//reset terminals
			_.each(terminals, function(el, i, arr){
				el.scaling = new paper.Point(1.0, 1.0);
			})
		},
		onMouseMove: function(event, hitResult, scope){
			if(!_.isNull(scope.selectedPath)){
				OhmTool.glow(scope.selectedPath, false);
			}
		}
	}
}


OhmTool.glow = function(path, isOn){
	// console.log(path);
	if(_.isNull(path)){
		console.log("Glow path is null");
		return;
	}
	if(["CP:_circuit_x5F_led_1_", "CP:_Battery"].indexOf(path.name) > -1){
		if(isOn){
			path.style = {
				shadowColor: "blue",
				shadowBlur: 30,
				shadowOffset: new paper.Point(0, 0)
			}
			path.opacity = 1;
		}
		else{
			path.style = {
				shadowColor: "blue",
				shadowBlur: 0,
				shadowOffset: new paper.Point(0, 0)
			}
		}
	}
	else if(path.className == "Group"){
		_.each(path.children, function(el, i, arr){
			OhmTool.glow(el, isOn);
		});
		if(isOn)path.opacity = 1;
		else path.opacity = 0;
	}
	else{
		if(isOn){
			path.style = {
				shadowColor: "blue",
				shadowBlur: 30,
				shadowOffset: new paper.Point(0, 0)
			}
			path.opacity = 1;
		}
		else{
			path.style = {
				shadowColor: "blue",
				shadowBlur: 0,
				shadowOffset: new paper.Point(0, 0)
			}
			if(path.Bender_path) path.opacity = 0;
		}
	}
}





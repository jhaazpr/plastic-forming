CircuitLayer.POSITIVE = new paper.Color("red");
CircuitLayer.NEGATIVE = new paper.Color("black");
CircuitLayer.NEUTRAL = new paper.Color("#CCC");
// CircuitLayer.NEUTRAL.saturation -= 0.5;
CircuitLayer.scaleable = false;
CircuitLayer.translateable = true;
CircuitLayer.rotateable = true;
CircuitLayer.silver = new paper.Color("#839CA5");
CircuitLayer.legend = function(parent, pt){
	legend = new paper.Group({
		parent: parent,
		name: "NC: artboard"
	});
	var positive =  new paper.Path.Rectangle({
		parent: legend, 
		fillColor: CircuitLayer.POSITIVE, 
		width: 15, 
		height: 15, 
		// strokeColor: "black", 
		strokeWidth: 0.5,
		position: new paper.Point(0, 0)
	});
	var ptext = new paper.PointText({
			parent: legend, 
			point: new paper.Point(0, 00),
			content: "POSITIVE",
			fillColor: 'black', 
			fontFamily: 'Arial', 
			// fontWeight: 'bold', 
			fontSize: 12
	});
	ptext.point.x += ptext.bounds.width/4;
	ptext.point.y += ptext.bounds.height/3;
	var negative =  new paper.Path.Rectangle({
		parent: legend, 
		fillColor: CircuitLayer.NEGATIVE, 
		width: 15, 
		height: 15, 
		// strokeColor: "black", 
		strokeWidth: 0.5,
		position: new paper.Point(0, 20)
	});
	var netext = new paper.PointText({
			parent: legend, 
			point: new paper.Point(0, 20),
			content: "NEGATIVE",
			fillColor: 'black', 
			fontFamily: 'Arial', 
			// fontWeight: 'bold', 
			fontSize: 12
	});
	netext.point.x += netext.bounds.width/4;
	netext.point.y += netext.bounds.height/3;
	var neutral =  new paper.Path.Rectangle({
		parent: legend, 
		fillColor: CircuitLayer.NEUTRAL, 
		width: 15, 
		height: 15, 
		// strokeColor: "black", 
		strokeWidth: 0.5,
		position: new paper.Point(0, 40)
	});
	var ntext = new paper.PointText({
			parent: legend, 
			point: new paper.Point(0, 40),
			content: "NEUTRAL",
			fillColor: 'black', 
			fontFamily: 'Arial', 
			// fontWeight: 'bold', 
			fontSize: 12
	});
	ntext.point.x += ntext.bounds.width/4;
	ntext.point.y += ntext.bounds.height/3;

	var g =  new paper.Path.Rectangle({
		parent: legend, 
		fillColor: "#00A8E1", 
		width: 15, 
		height: 15, 
		// strokeColor: "black", 
		strokeWidth: 0.5,
		position: new paper.Point(0, 60)
	});
	var gtext = new paper.PointText({
			parent: legend, 
			point: new paper.Point(0, 60),
			content: "GUIDES",
			fillColor: 'black', 
			fontFamily: 'Arial', 
			// fontWeight: 'bold', 
			fontSize: 12
	});
	gtext.point.x += gtext.bounds.width/4;
	gtext.point.y += gtext.bounds.height/3;


	var g =  new paper.Path.Rectangle({
		parent: legend, 
		rectangle: legend.bounds.expand(15), 
		fillColor: "white", 
		radius: 5
	});
	g.sendToBack();

	legend.position = pt;
	legend.position.x -= legend.bounds.width + 50;
	legend.position.y += 100;
	legend.scaling.x = paper.view.zoom;
	legend.scaling.y =  paper.view.zoom;
	return legend;
}
function CircuitLayer(paper, parent, material){
	this.paper = paper;
	this.className = "CircuitLayer";

	this.layer = new paper.Layer({
		name: material + ":" + "CircuitLayer"
	});
	this.layer.remove();
	parent.addChild(this.layer);
	var legendPosition = paper.view.bounds.topRight.clone();
	this.legend = CircuitLayer.legend(this.layer, legendPosition);
	this.legend.remove();
	this.connections;
}

CircuitLayer.prototype = {
	add: function(layer, single){
		var scope = this;
		if(single){
			layer.remove();
			layer.layerClass = scope.className;
			
			var cp = new paper.Group({
				parent: scope.layer,
				name: "CP: Added trace", 
				layerClass: scope.className, 
				children: [layer], 
				canvasItem: true
			});

			this.circuit_view();
		}
		else{
			_.each(layer, function(el, i, arr){
				el.remove();
				el.layerClass = scope.className;
				scope.layer.addChild(el);
			});
			this.trace_view();
		}
	},
	print_view: function(){
		this.legend.remove();
		CircuitLayer.select_and_color_and_code(this.layer, ["CNP", "CGP", "CVP"], { strokeColor: "#ED1E79", shadowBlur: 0, dashArray:[10, 1]});
		CircuitLayer.select_and_color_and_code(this.layer, ["CNT", "CGT", "CVT", "CVTB", "CGTB"], { fillColor: "#ED1E79", shadowBlur: 0, dashArray: [10, 1]});

		paper.view.update();
	},
	trace_view: function(){
		this.legend.remove();
		CircuitLayer.select_and_color_and_code(this.layer, ["CNP", "CGP", "CVP"], 
			{ 
				strokeColor:CircuitLayer.silver, 
				dashArray:[], 
				shadowColor: new paper.Color(0.8),
			  	shadowBlur: 0,
			  	shadowOffset: new paper.Point(0, 0)
			});

		// console.log("CVT", BenderSVG.match(this.layer, {prefix: ["CVT", "CGT", "CNT"]}));
		CircuitLayer.select_and_color_and_code(this.layer, ["CNT", "CGT", "CVT", "CVTB", "CGTB"], 
			{ fillColor: CircuitLayer.silver, 
			  dashArray:[], 
			  shadowColor: new paper.Color(0.8),
			  shadowBlur: 0,
			  shadowOffset: new paper.Point(0, 0)
			});


		paper.view.update();
	},
	connect_view: function(){
		
		var elements = BenderSVG.match(this.layer, { prefix: ["CNP", "CGP", "CVP"] });
		style = {strokeColor: "black", strokeWidth: 1}
		this.pre_connection = _.map(elements, function(el, i, arr){
			prevColor = el.style.strokeColor;
			prevWidth = el.style.strokeWidth;
			el.style = style;
			// el.polarity = polarity;
			return {el: el, strokeColor: prevColor, strokeWidth: prevWidth}
		});

		// ALL CONDUCTIVE PATHS
		var conductive = ["CGP", "CNP", "CVP", "CGB", "CVB", "CNB", "CGT", "CVT", "CNT", "CVTB", "CGTB"];
		var conductive = BenderSVG.get({ prefix: conductive });
		var conductiveIDs = _.map(conductive, function(el){ return el.id; });
		
		// GET PAIR TRAVERSAL
		var pairs = _.map(conductiveIDs, function(id){
			return _.compact(_.map(conductiveIDs, function(id2){
				if(id == id2) return false;
				return _.sortBy([id, id2]);
			}));
		});
		pairs = _.flatten(pairs, true);
		pairs =  _.uniq(pairs, function(el){
			return el[0] * 100 + el[1];
		});
		clusters = _.groupBy(pairs, function(el){
			return el[0];
		});
		this.connections = new paper.Group({
			name: "NC: Connections", 
			terminal_helper: true
		})
		var scope = this;
		_.each(clusters, function(cluster, key){
			condA = Node.get(parseInt(key));
			comps = _.map(cluster, function(el, i){
				return Node.get(el[1]);
			});
			ixts = TracePathTool.getAllIntersections(condA, comps);
			// console.log(key, ixts.length);
			_.each(ixts, function(el, i, arr){
				var c = new paper.Path.Circle({
					parent: scope.connections,
					radius: 4, 
					fillColor: "blue", 
					position: el.point, 
					strokeColor: "#00A8E1", 
					strokeWidth: 1
				});
				// c.remove();
				// scope.connections.addChild(c);
			});
		});
		// this.connections.remove();
		paper.view.update();
	},
	unconnect_view: function(){
		this.connections.remove();

		 _.each(this.pre_connection, function(item, i, arr){
		 	// console.log(item);
			item.el.style.strokeColor = item.strokeColor;
			item.el.style.strokeWidth = item.strokeWidth;
			// item.el.style.strokeColor = item.style.strokeColor;
			// item.el.polarity = polarity;
		});
		paper.view.update();
		 

	},	
	resetLegend: function(){
		
		this.legend.position = artboard.bounds.topRight.clone();
		this.legend.position.x -= legend.bounds.width + 50;
		this.legend.position.y += 100;
		// this.legend.scaling.x /= paper.view.zoom;
		// this.legend.scaling.y /= paper.view.zoom;
	},
	circuit_view: function(){
		this.addLegend();
		// this.legend.scaling = new paper.Point(this.paper.view.zoom, this.paper.view.zoom);
		// paper.view.bounds.topRight.clone();
		CircuitLayer.select_and_color_and_code(this.layer, ["CNT"], { shadowBlur: 0, fillColor: CircuitLayer.NEUTRAL}, CircuitLayer.NEUTRAL);
		CircuitLayer.select_and_color_and_code(this.layer, ["CNP"], { shadowBlur: 0, strokeColor: CircuitLayer.NEUTRAL}, CircuitLayer.NEUTRAL);
		CircuitLayer.select_and_color_and_code(this.layer, ["CVT", "CVTB"], { shadowBlur: 0, fillColor: CircuitLayer.POSITIVE}, CircuitLayer.POSITIVE);
		CircuitLayer.select_and_color_and_code(this.layer, ["CVP"], { shadowBlur: 0, strokeColor: CircuitLayer.POSITIVE}, CircuitLayer.POSITIVE);
		CircuitLayer.select_and_color_and_code(this.layer, ["CGT", "CGTB"], { shadowBlur: 0, fillColor: CircuitLayer.NEGATIVE}, CircuitLayer.NEGATIVE);
		CircuitLayer.select_and_color_and_code(this.layer, ["CGP"], { shadowBlur: 0, strokeColor: CircuitLayer.NEGATIVE}, CircuitLayer.NEGATIVE);

		paper.view.update();
	}, 
	addLegend: function(){
		this.layer.addChild(this.legend);
		this.resetLegend();
	}

}


CircuitLayer.select_and_color_and_code = function(collection, prefixes, style, polarity){
	var elements = BenderSVG.match(collection, { prefix: prefixes });
	_.each(elements, function(el, i, arr){
		el.style = style;
		el.polarity = polarity;
	});
}
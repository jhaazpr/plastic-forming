
function CanvasItem(paper, path, type){
	this.className = "CanvasItem";
	this.paper = paper;
	this.id = path.id;
	this.guid = guid();
	
	this.type = type;
	
	if(type == "ArtworkLayerElement")
		this.path = path;
	else
		this.path = new paper.Group([path]);
	
	this.path.canvasItem = this;
	this.path.applyMatrix = true;
	this.name = path.name;

	// transformation rectangle
	this.ref_x = false;
	this.ref_y = false;

	this.circuit_label = "";
}
CanvasItem.prototype = {
	setOpacity: function(val){
		this.path.opacity = val;
	},
	getBounds: function(){
		return this.path.bounds;
	},
	duplicate: function(){
		var p = this.path.clone();
		var wp = new CanvasItem(paper, p);
		wp.material = this.material;
		wp.reflect_x = this.reflect_x;
		wp.reflect_y = this.reflect_y;
		wp.path.id = guid();
		wp.path.position.x +=5;
		wp.update();
		return wp;
	},
	reflect_x: function(){
		this.ref_x != this.ref_x;
		this.path.scaling.x *= -1;
		this.paper.view.update();
	},
	reflect_y: function(){
		this.ref_y != this.ref_y;
		this.path.scaling.y *= -1;
		this.paper.view.update();
	},
	remove: function(){
		this.path.remove();
	}
}



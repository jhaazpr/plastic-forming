Node.touch = function(id){
	var n = Node.get(id);
	n.selected = !n.selected;
}
function Node(paths, position){
	var scope = this;
	this.className = "Node";
	this.paths = paths;
	var polarity = TracePathTool.readPolarity(paths[0]);
	// console.log("I am here");
	this.parent = paths[0].parent;
	var c_int = new paper.Path.Circle({
					parent: paths[0].parent,
					// parent: paper.project.layers 
					paths: paths,
					name: "TMT: temp", //"C"+ polarity +"T: terminal from fragmentation",
					position: position,
					radius:  paths[0].style.strokeWidth, 
					fillColor:  "#00A8E1", 
					polarity:  paths[0].polarity,
					terminal_helper: true,
					is_node: true,
					strokeColor: "black", 
					strokeWidth: 1, 
					node: this // pointer to self
				});
	
	this.self = c_int;
	this.id  = c_int.id;
	_.each(paths, function(path){
		if(_.isUndefined(path.terminals)) path.terminals = [scope.self.id];
		else path.terminals.push(scope.self.id);
	});
	this.children = [];
}

Node.centroid = function(nodes){
	centroid = new paper.Point(0, 0);
	_.each(nodes, function(el){
		centroid.x += el.self.position.x; 
		centroid.y += el.self.position.y; 
	});
	centroid.x /= nodes.length;
	centroid.y /= nodes.length;
	return centroid;
}
Node.allChildren = function(nodes, nodeIDs){
	var children = _.reduce(nodes, function(memo, el){
		memo.push(el.getChildren())
		return memo;
	}, []);
	children = _.uniq(_.flatten(children));
	children = _.filter(children, function(child){
		return nodeIDs.indexOf(child) < 0;
	});
	return children;
}
Node.toPaths = function(ids){
	return _.map(ids, function(node){
		var n = Node.get(node);
		return n;
	});
}
Node.toNodes = function(nodeIDs){
	return _.map(nodeIDs, function(node){
		var n = Node.get(node);
		if(!n) return n;
		if(_.isUndefined(n.node)){
			console.error("Attempted to extract node from", n);
			return null;
		}
		return n.node;
	});
}
Node.toNodeIDs = function(nodes){
	return _.map(nodes, function(node){
		return node.id;
	});
}
Node.get = function(id){
	var n =	paper.project.getItem({id: id});
	if(_.isNull(n)){
		console.error("Attempted to Node.get", id);
	}
	return n;
}
Node.isValid = function(id){
	return !_.isNull(Node.get(id));
}
Node.join = function(nodeIDs){
	// console.log("JOIN", nodeIDs);
	var nodeIDs = _.filter(nodeIDs, function(id){
		return Node.isValid(id);
	});
	if(nodeIDs.length == 0) return null;
	var nodes = Node.toNodes(nodeIDs)
	
	var centroid = Node.centroid(nodes);
	var children = Node.allChildren(nodes, nodeIDs);


	// For all the ones that are about to be deleted, 
	// go through the children, remove references to them. 
	_.each(children, function(id){
		var node = Node.get(id).node;
		var children = node.getChildren();
		children = _.reject(children, function(child){
			return nodeIDs.indexOf(child) > -1;
		});
		node.setChildren(children);
	});
	
	// Remove old path index


	updated_paths = [];
	_.each(nodes, function(node){
		var paths = node.paths;
		updated_paths.push(paths);
		_.each(paths, function(path){
			path.terminals = _.reject(path.terminals, function(term){
				return nodeIDs.indexOf(term) > -1;
			});
		});
	});
	updated_paths =_.flatten(updated_paths);
	updated_paths =_.uniq(updated_paths, function(el){
		return el.id;
	});

	return {position: centroid, paths: updated_paths, children: children};
}


Node.prototype = {
	disable: function(){
		this.self.remove();
	},
	enable: function(){
		// if(_.isUndefined(this.self.homed)){
			// this.self.homed = true;
			// this.self.position.x -= 145;
		// }
		this.parent.addChild(this.self);
		this.self.parent.bringToFront();
	},
	getPosition: function(){
		var pos = this.self.position;
		if(_.isUndefined(pos)){
			this.enable();
			pos = this.self.position;
			this.disable();
		}
		return pos;
	},
	setChildren: function(children){
		this.children = children;
	},
	getChildren: function(){
		scope = this;
		return _.reject(this.children, function(el){ return scope.id == el});
	}, 
	colorize: function(color){
		if(TracePathTool.isPath(this.self)) this.self.style.strokeColor = color;
		else this.self.style.fillColor = color;
		paper.view.update();
		return this;
	}
}

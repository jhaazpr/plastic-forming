
var EPSILON = 2;
solutions = null;


function Graph(){
	this.className = "Graph";
	this.nodes = [];
	this.breakUpSelfIntersections();
	this.breakUpIntersections();
	this.generateBlobPaths();

	// this.colorPaths();
	this.init();
	
}


var intersectionNodes = [];
Graph.prototype = {
	colorPaths: function(){
		var hue = 0;

		var traces = ["CGP", "CVP", "CNP", "TGP", "TVP", "TNP"];

		traces = BenderSVG.match(paper.project, { prefix: traces });

		for(var i in traces){
			var trace = traces[i];
			c = new paper.Color("red");
			c.hue = hue;
			trace.strokeColor = c;
			hue += 20;
		}
	},
	init: function(){
		this.processEnds();
		this.blobConsolidation();
		this.lines = $.map(paper.project.getItems({blob_line: true}), function(el, i){
			parent = el.parent;
			el.remove();
			return {el: el, parent: parent}
		});

	},
	find: function(id){
		return _.filter(graph.nodes, function(el, i, arr){ return el.id == id})[0];
	},
	enable: function(){
		_.each(this.nodes, function(node){
			node.enable();
			// node.self.position.x -= 145/6;
		});

		paper.view.update();
	},
	disable: function(){
		_.each(this.nodes, function(node){
			node.disable();
		});
		paper.view.update();
	},
	processEnds: function(){
		var scope = this;

		var traces = ["CGP", "CVP", "CNP", "TGP", "TVP", "TNP"];
		traces = BenderSVG.match(paper.project, { prefix: traces });
		
		var nodes = [];

		// add nodes to the ends
		_.each(traces, function(trace, i, arr){
				// console.log("finished", i, arr.length);
				// console.log(trace);
			if(trace.length != 0){
				var start = trace.getPointAt(0).clone();
				var end = trace.getPointAt(trace.length).clone();
				
				n = new Node([trace], start);
				nodes.push(n);
				scope.addNode(n);
				// console.log("NODE", n.self.id);

				n = new Node([trace], end);
				nodes.push(n);
				scope.addNode(n);
				// console.log("NODE", n.self.id);
			}
		});

		
		// EXTRACT EDGES
		var nodeSet = _.map(nodes, function(n){ return n.self });
		_.each(nodes, function(node, i, arr){
			var nodeC = node.self;
			var edges = scope.getNodeIntersections(node.self.id);
			// console.log(nodeC.id, "P_EDGES", edges)
			// compare = _.reject(nodeSet, function(n){ return n.id == nodeC.id});
			// var edges = TracePathTool.getAllIntersections(nodeC, compare);
			// edges = _.map(edges, function(edge, i, arr){ return edge._curve2.path.id});
			// var pathID = nodeC.parent.id;
			
			terminals = _.reduce(nodeC.paths, function(memo, path){
				memo.push(path.terminals);
				return memo;
			}, []);
			terminals = _.flatten(terminals);

			pathBrother = _.reject(terminals, function(term){
				return nodeC.id == term;
			});
			edges = _.flatten([pathBrother, edges]);
			// console.log(nodeC.id, "EDGES", edges);
			nodeC.node.setChildren(_.uniq(edges));
		});

		// IDENTIFY CLUSTER NODES
		// console.log("CLUSTERS");
		groups = _.map(nodes, function(node, i, arr){
			var nodeC = node.self;
			var edges = scope.getNodeIntersections(node.self.id);
			edges.push(nodeC.id);
			edges = _.sortBy(edges);
			// console.log(edges);
			return edges;
		});

		groups = _.groupBy(groups);
		// groups = 
		_.each(groups, function(value, key){
			var ids = key.split(",");
			for(var i in ids) ids[i] = parseInt(ids[i]);
			// return ids;
		});

		groups2 = {};
		var getOverlaps = function(i, keys){
			return _.filter(keys, function(el){
				if(el == i) return;
				a = el.split(',');
				b = i.split(',');
				return _.intersection(a, b).length > 3;
			});
		}
		var keys = _.keys(groups);
		for(var key in groups){
			overlaps = getOverlaps(key, keys);
			// console.log(overlaps);
			if(overlaps.length > 0){
				var key = _.reduce(overlaps, function(memo, overlap){
					return _.intersection(memo, overlap.split(','));
				}, key.split(','))
				groups2[key.join(',')] = true;
			}
			else if(key.split(',').length > 3){
				groups2[key] = true;
			}
		}
		groups2 = _.map(_.keys(groups2), function(key){
			return _.map(key.split(','), function(el){
				return parseInt(el);
			});
		});
		// console.log("END CLUSTERS");



		// // CONSOLIDATE nodes
		_.each(groups2, function(g, i, arr){
			// if(i > 2) return;
			
			// console.log(g);
			// console.log(i.split(","));
			test = Node.join(g);
			if(!_.isNull(test)){
				n = new Node(test.paths, test.position);
				nodes.push(n);
				// console.log(g, "-->", n.id);
				
				scope.addNode(n);
				

				test.children.push(n.id);
				_.each(test.children, function(child, i, arr){
					// console.log("HMM", child, n.id);
					var x = Node.get(child).node;
					// console.log(n.id);
					x.children.push(n.id);
				});
				scope.removeNodes(g);
				n.setChildren(test.children);
			}
		});

		// console.log("NODE", n.self.id);
	},
	removeNodes: function(removeThese){
		this.nodes = _.reject(this.nodes, function(node){
			
			var marked = removeThese.indexOf(node.id) > -1
			if(marked){
				// console.log("DELETING", node.id)
				node.self.remove();
				delete node;
			}
			return marked;
		});
	},
	getNodeIntersections: function(nodeID){
		var nodePath = Node.get(nodeID);
		var nodeCs = paper.project.getItems({is_node: true});

		nodeCs = _.reject(nodeCs, function(n){ return n.id == nodeID});

		// if(nodeID == 70) console.log("NODECs 70:", nodeCs);

		inside = _.filter(nodeCs, function(el, i, arr){
			return nodePath.position.isInside(el.bounds) || el.position.isInside(nodePath.bounds);
		});
		inside = _.map(inside, function(el, i, arr){
			return el.id;
		});
		nodeCs = _.map(nodeCs, function(el, i, arr){
			return el.id;
		});

		return inside;
	},
	generateBlobPaths: function(){
		var scope = this;
		var blobs = ["CGB", "CVB", "CNB", "CGT", "CVT", "CNT", "CNTB", "CVTB", "CGTB"];
		blobs = BenderSVG.match(designer.circuit_layer.layer, { prefix: blobs });
		// console.log("Blob count", blobs.length);
		var traces = ["CGP", "CVP", "CNP"];		
		traces = BenderSVG.match(designer.circuit_layer.layer, { prefix: traces });
		
		// PATH SPLITTING IN BLOBS
		var path_processing = _.map(blobs, function(blob, i, arr){
			blob.paths = [];
			intersects = TracePathTool.getAllIntersections(blob, traces);
			p_offsets = _.map(intersects, function(its){
				var path = its._curve2.path;
				var pt = path.getNearestPoint(its.point);
				var offset = path.getOffsetOf(pt);
				return {path: path.id, offset: offset, source: its._curve.path.id}
			});
			if(intersects.length == 0){
				var near = blob.position.clone();
				near.y -= 5;
				near.x += 5;
				b = TracePathTool.readPolarity(blob);
				var path = new paper.Path({
					blob: blob.id,
					name: "T"+b+"P: temporary",
					segments: [blob.position.clone(), near], 
					strokeColor: "yellow", 
					strokeWidth: 2,
					terminal_helper: true
				});
				blob.paths.push(path);
			}	
			return p_offsets;
		});

		p_offsets = _.flatten(path_processing);
		p_offsets = _.groupBy(p_offsets, function(el){ return el.path; });

		// console.log("HMM", p_offsets);
		created = _.map(p_offsets, function(value, pathID){
			var pathID = parseInt(pathID);
			var offsets = _.map(value, function(el){ return el.offset; });
			var sources = _.map(value, function(el){ return el.source; });
			var newPaths = scope.splitPathAtOffsets(Node.get(pathID), offsets);
			sources = _.sortBy(sources, function(el, i){ return offsets[i]; });
			return { newPaths: _.flatten([pathID, newPaths]), sources: _.flatten([sources[0], sources]) }
		});
		created = _.flatten(created);
		var hue = 0;

		var bound_check = function(position, blob){
			var c = new paper.Path.Circle({
				radius: 5, 
				position: position, 
				strokeColor: "red", 
				strokeWidth: 1
			});
			ixts = _.map(blobs, function(el, i){
				if(c.intersects(el))
					return el.id;
			});
			
			c.remove();
			return _.compact(ixts);
		} 


		var createBlobPath = function(sourceBlob, pos){
			
			b = TracePathTool.readPolarity(sourceBlob);
			var center = sourceBlob.position.clone();
			var newPath = new paper.Path({
				blob: sourceBlob.id,
				name: "T"+ b +"P: temporary",
				segments: [center, pos], 
				strokeColor: "yellow", 
				strokeWidth: 2,
				terminal_helper: true, 
				blob_line: true
			});
			return newPath;
		}
		var pathify = function(sourceBlob, path){
			var start = path.getPointAt(0);
			var end = path.getPointAt(path.length);

			var ixts_start = bound_check(start, sourceBlob)
			var ixts_end = bound_check(end, sourceBlob)
			// console.log(path.id, ixts_start, ixts_end);
			_.each(ixts_start, function(el){
				var sourceBlob = Node.get(el);
				newPath = createBlobPath(sourceBlob, start);
				sourceBlob.paths.push(newPath);
			});
				
			_.each(ixts_end, function(el){
				var sourceBlob = Node.get(el);
				newPath = createBlobPath(sourceBlob, end);
				sourceBlob.paths.push(newPath);
			});
			
			
		}
		_.each(created, function(path_created, i){
				groups = _.groupBy(path_created.newPaths, function(el, j){
					return path_created.sources[j];
				});
				_.each(groups, function(group, j){
					var source = Node.get(parseInt(j));
					for(var i in group){
						var path = Node.get(group[i]);
						if(source.contains(path.getPointAt(path.length/2))){
							path.remove();							
						}else{
							pathify(source, path);
						}
					}	
				});	
		});

		
			
		
		
	},
	blobConsolidation: function(){
		var scope = this;
		var blobs = ["CGB", "CVB", "CNB", "CGT", "CVT", "CNT", "CNTB", "CVTB", "CGTB"];
		blobs = BenderSVG.match(designer.circuit_layer.layer, { prefix: blobs });
		
		_.each(blobs, function(blob){
			g = _.map(blob.paths, function(p){
				return p.terminals[0];
			});
			// console.log(blob.id, g);
			newNode = Node.join(g);
			if(!_.isNull(newNode)){
				n = new Node(newNode.paths, newNode.position);
				blob.sourceNode = n.id;
				// nodes.push(n);
				// console.log(g, "-->", n.id);
				
				scope.addNode(n);
				

				newNode.children.push(n.id);
				_.each(newNode.children, function(child, i, arr){
					// console.log("HMM", child, n.id);
					var x = Node.get(child).node;
					// console.log(n.id);
					x.children.push(n.id);
				});
				scope.removeNodes(g);
				n.setChildren(newNode.children);
			}
			
		});
	},
	breakUpIntersections: function(){
		var scope = this;
		var conductiveTraces = ["CGP", "CVP", "CNP"];
		conductiveTracesA = BenderSVG.match(designer.circuit_layer.layer, { prefix: conductiveTraces });
		cuts = [];
		_.each(conductiveTracesA, function(el, i, arr){
			el.self = true;	
			conductiveTracesB = BenderSVG.match(designer.circuit_layer.layer, { prefix: conductiveTraces });
			conductiveTracesB = _.reject(conductiveTracesB, function(el2, i2, arr2){
				return el2.self || el2.processed;
			});
			// console.log("Intersecting", el.id)
			intersects = TracePathTool.getAllIntersections(el, conductiveTracesB);
			// console.log("Exiting", el.id)
			el.processed = true;

			_.each(intersects, function(el2, i2, arr2){
				var pathIn = el2._curve.path;
				var pathOut = el2._curve2.path;
				var point = el2.point.clone();
				var pointA = pathIn.getNearestPoint(point);
				var offsetA = pathIn.getOffsetOf(pointA);
				var pointB = pathOut.getNearestPoint(point);
				var offsetB = pathOut.getOffsetOf(pointB);

				if(offsetA && scope.isValidSplitLocation(pathIn, offsetA))
					cuts.push({point: point.clone(), path: pathIn.id, offset: offsetA});
				if(offsetB && scope.isValidSplitLocation(pathOut, offsetB))
					cuts.push({point: point.clone(), path: pathOut.id, offset: offsetB});
					
			});
			el.self = false;		
		});
		
		cuts = _.groupBy(cuts, function(el, i, arr){
			return el.path;
		});

		// CUT IT OPEN
		_.each(cuts, function(el, pathID, arr){
			var pathID = parseInt(pathID);
			var offsets = _.sortBy(_.pluck(el, "offset"));
			// console.log(pathID, offsets)
			scope.splitPathAtOffsets(Node.get(pathID), offsets);
			
		});

		// console.log("CUTS", cuts);

	},
	isValidSplitLocation: function(path, offset){
		return offset > EPSILON && path.length - offset > EPSILON;
	}, 
	splitPathAtOffsets: function(path, offsets){
		// console.log("Splitting", path.id, "@", offsets, path.length);
		var scope = this;
		offsets = _.sortBy(offsets);
		offsets = _.filter(offsets, function(e){
			return scope.isValidSplitLocation(path, e);
		});
		offsets = _.uniq(offsets);
		// console.log('Slicing', path.id, "@", offsets, path.length);

		var cut = 0;
		var newPath = path;
		var created = [];

		_.each(offsets, function(offset){
			// console.log(offset, offset - cut);
			newPath = newPath.split(offset - cut);
			created.push(newPath.id);
			cut += (offset - cut);
		});
		// console.log("Created:", created);
		return created;
	},
	breakUpSelfIntersections: function(){
		var scope = this;
		var traces = ["CGP", "CVP", "CNP"];

		traces = BenderSVG.match(designer.circuit_layer.layer, { prefix: traces });
		_.each(traces, function(el, i, arr){
			
			self_int = el.getIntersections(el);

			_.each(self_int, function(el2, i2, arr2){
				var path = el2._curve.path;
				var point = el2._point;
				var offsetBeforeLoop = path.getOffsetOf(point);
				var loop = path.split(offsetBeforeLoop);
				
				if(loop){
						black_magic = loop.clone();
						black_magic_2 = black_magic.split(1);
						if(black_magic_2){
							offsetPostLoop = black_magic_2.getOffsetOf(point);
							black_magic_2.remove();
							black_magic.remove();
							// end of magic
							var end_of_path = loop.split(offsetPostLoop);
							if(end_of_path){
								start_of_path = path;
								var ids = [start_of_path.id, end_of_path.id, loop.id]
							}
						}
					}

				});

			});
	},
	

	addNode: function(node){
		this.nodes.push(node);
	},
	getSourceNode: function(){
		var batt_terminals = BenderSVG.match( designer.circuit_layer.layer, { prefix: ["CVTB"]});
		if(batt_terminals.length > 0) return batt_terminals[0].id;
		else return null;
	},
	getSinkNode: function(){
		var batt_terminals = BenderSVG.match( designer.circuit_layer.layer, { prefix: ["CGTB"]});
		if(batt_terminals.length > 0) return batt_terminals[0].id;
		else return null;
	},
	
	update: function(){
		paper.view.update();
		return this;
	}
}




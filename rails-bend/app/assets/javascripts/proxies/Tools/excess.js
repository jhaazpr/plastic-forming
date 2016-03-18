<div id="artwork-toolbox" class="toolbox shadowed">
   <label class="text-shadowed"> ARTWORK </label>
   <div class="btn-group" role="group">
     
        <%= button_tag :id=>"lock", :class=>"btn btn-md btn-ellustrator" do %>
          <span class="icon-lock2"></span>
        <% end %>
         <%= button_tag :id=>"ghost", :class=>"btn btn-md btn-ellustrator" do %>
          <span class="icon-ghost"></span>
        <% end %>
   </div>
</div>



   <!-- ZOOM TOOLS -->
      
      <% button_tag :id=>"scale", :class=>"btn btn-md btn-ellustrator" do %>
          <span class="icon-ruler"></span>
        <% end %>
     
      <!-- END ZOOM TOOLS -->

<!-- </div> -->
 <% button_tag :id => "anchor-tool", :class => "btn btn-lg btn-ellustrator" do %>
        <span class="icon-pen-tool"></span>
        <% end %>

      <% button_tag :class => "btn btn-ellustrator" do %>
        <span class="icon-noun_68342_cc"></span>
      <% end %>
      <% button_tag :class => "btn btn-ellustrator" do %>
        <span class="icon-bezier-select"></span>
      <% end %>
      <% button_tag :class => "btn btn-ellustrator" do %>
        <span class="icon-anchor-select"></span>
      <% end %>

  <div class="btn-group" role="group" aria-label="pen-tools">
   <% button_tag :id => "run-tool", :class => "btn btn-primary" do %>
    <span class="glyphicon glyphicon glyphicon-record"></span>
    <% end %>
      <% button_tag :id => "ohm-tool", :class => "btn btn-primary" do %>
    Ω
    <% end %>
     <% button_tag :id => "fab-tool", :class => "btn btn-primary" do %>
    <span class="icon-inkwell"></span>
    <% end %>
    <% button_tag :id => "debug-tool", :class => "btn btn-primary" do %>
    <span class="icon-debug"></span>
    <% end %>
  </div>   




onTap: function(event, hitResult, scope){
			// console.log("hello!", event, event.point);
			scope.canvas_item_type = null;
			var pos = scope.paper.view.viewToProject(new paper.Point(event.center.x, event.center.y));
			// check for bogus taps 
			var c  = paper.Path.Circle({
								radius: HIT_TEST_BOUNDING_RADIUS, 
								fillColor: "red", 
								position: pos
							});

			var paths = BenderSVG.match(paper.project, {className: "Path"})
			var intersections = []; 
			for(var i in paths){
				var a = c.getIntersections(paths[i]);
				if(a.length > 0)
					intersections.push(a);
			}
			intersections = _.flatten(intersections);

			if(intersections.length > 0){
				_.each(intersections, function(el, i, arr){
					var cluster = el._curve2.path;
					while(_.isUndefined(cluster.canvasItem))
						cluster = cluster.parent;
					scope.sm.add(cluster, event.srcEvent.shiftKey);
				});
			}
			else{
				scope.sm.clear();
			}
			c.remove();
		},


		regenerate: function(){
		console.log("Regenerating!");
		this.enable();
		nodeIDs = Node.toNodeIDs(this.nodes);
		Node.join(nodeIDs);
		this.removeNodes(nodeIDs);
		// END OF DESTRUCTION
		
		_.each(this.lines, function(line){
				line.parent.addChild(line.el);
		});
		this.init();
		paper.view.update();
	}, 

<nav class="navbar navbar-default shadowed" role="navigation">
	  <div class="container-fluid">
	    <!-- Brand and toggle get grouped for better mobile display -->
	    <div class="navbar-header">
	      <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1">
	        <span class="sr-only">Toggle navigation</span>
	        <span class="icon-bar"></span>
	        <span class="icon-bar"></span>
	        <span class="icon-bar"></span>
	      </button>
	      <a class="navbar-brand" href="/">
	      	<b id='logo'>
	      		<img src="/led.png"/>
	      	</b>
	      </a>
	      <div id="design-name" class="navbar-brand" href="">
	      	<b contentEditable="true"> </b>
	      </span>
	      </div>
	    </div>

	    <!-- Collect the nav links, forms, and other content for toggling -->
	    <div class="collapse navbar-collapse" >
	    	<div id="main-actions">
		  <%= render :partial => "usernav.html.erb" %>

	      <ul id="history-bar" class="nav navbar-nav pull-right">
	      	    <li>
		        	<% link_to "javascript:void(0)", :id=>"revert", :class=>"" do %>
		          		<span class="glyphicon glyphicon-refresh"></span>
		        		Revert
		        	<% end %>
		        </li>
		        <li>
		        	<%= link_to "javascript:void(0)", :id=>"undo", :class=>"" do %>
		          		<span class="icon-undo"></span>
		        	<% end %>
		        </li>
		      
		        <li>
		        	<%= link_to "javascript:void(0)", :id=>"redo" do %>
		          		<span class="icon-redo"></span>
		        	<% end %>
		        </li>

		        <li>
		        	<%= link_to "javascript:void(0)", :id=>"save-json", :class=>"" do %>
		          		<span class="glyphicon glyphicon-floppy-disk"></span>
		        	<% end %>
		        </li>
		        <!-- <li>
		        	<%= link_to :id=>"fast-forward", :class=>"btn btn-md btn-ellustrator" do %>
		          		<span class="glyphicon glyphicon-repeat"></span>
		        	<% end %>
		        </li> -->
	       </ul>
	      </div>
	    </div><!-- /.navbar-collapse -->
	    
	  </div><!-- /.container-fluid -->
	</nav>

		isValidNodeLocation: function(position){
		
		var loc = [];
		var valid =  _.reduce(intersectionNodes, function(memo, el){
					var d = el.point.getDistance(position);
					// console.log(el.self.id, d);
					if(d <= 3) loc.push(el.paths);
					return memo;//d > 3 && memo;
				}, true);
		return {valid: valid, paths: _.flatten(loc)}
	},
walkFrom: function(node){
		this.walked.push(node);
		node.walked = true;

		var children = node.getChildren();
		// rejected walked children
		children = _.reject(children, function(el, i, arr){ return el.walked;});
		// reject duplicates
		children = _.uniq(children, function(el, i, arr){ return el.self.id;});
		console.log("Node has", children.length, "children.");
		return children;		
	},

	processTraceIntersections: function(){
	var scope = this;
	var conductiveTraces = ["CGP", "CVP", "CNP"];
	conductiveTracesA = BenderSVG.match(designer.circuit_layer.layer, { prefix: conductiveTraces });
		_.each(conductiveTracesA, function(el, i, arr){
			el.self = true;	
			conductiveTracesB = BenderSVG.match(designer.circuit_layer.layer, { prefix: conductiveTraces });
			conductiveTracesB = _.reject(conductiveTracesB, function(el2, i2, arr2){
				return el2.self || el2.processed;
			});
			
			intersects = TracePathTool.getAllIntersections(el, conductiveTracesB);
			
			el.processed = true;
			_.each(intersects, function(el2, i2, arr2){
				var pathIn = el2._curve.path;
				var pathOut = el2._curve2.path;
				var offsetA = pathIn.getOffsetOf(el2.point);

				var position = pathIn.getPointAt(offsetA).clone();
				var ids = [pathIn.id, pathOut.id];
				
				// valid = scope.isValidNodeLocation(position);
				// if(valid)
					scope.addNode(new Node(pathIn, position, ids ));
				// console.log(valid);
				
				
			});

			el.self = false;		
		});
	},
processSelfIntersections: function(){
	var scope = this;
	var conductiveTraces = ["CGP", "CVP", "CNP"];
	conductiveTracesA = BenderSVG.match(designer.circuit_layer.layer, { prefix: conductiveTraces });
		_.each(conductiveTracesA, function(el, i, arr){
			self_int = el.getIntersections(el);
			_.each(self_int, function(el2, i2, arr2){
				var path = el2._curve.path;
				var pointA = el2._point;
				var offsetA = path.getOffsetOf(pointA);
				
				var loop = path.split(offsetA);
				// path.style.strokeColor = "green";
				// path.position.x -= 10;
				if(loop){
					// loop.style.strokeColor = "yellow";
					black_magic = loop.clone();
					black_magic_2 = black_magic.split(1);
					if(black_magic_2){
						black_magic_2.style.strokeColor = "red";
						offsetB = black_magic_2.getOffsetOf(pointA);
						black_magic_2.remove();
						black_magic.remove();
						// end of magic
						var end_of_path = loop.split(offsetB);
						if(end_of_path){
							// end_of_path.style.strokeColor = "blue";
							start_of_path = path;
							var ids = [start_of_path.id, end_of_path.id, loop.id]
							// var n = new Node(start_of_path, pointA, ids);
							intersectionNodes.push({point: pointA, paths: ids});
							// scope.addNode(n);
						}
					}
				}
			});

		});
	},// REDUDANCY CHECK

	// Initialize compute
	// var adj = {};
	// _.each(graph.nodes, function(el, i, arr){
	// 	adj[el.self.id] = [];
	// });

	// _.each(graph.nodes, function(el, i, arr){
	// 	var id = el.self.id;
	// 	var children = el.getChildren();
		
	// 	_.each(children, function(el2, i2, arr2){
	// 		adj[id].push(el2);
	// 		adj[el2].push(id);
	// 	});
	// });
	// _.each(adj, function(value, key, arr){
	// 	adj[key] = _.uniq(value);
	// });
	// _.each(graph.nodes, function(el, i, arr){
	// 	el.children = _.map(adj[el.self.id], function(el2, i2, arr2){ return paper.project.getItem({id: el2}).node; });;
	// });

	// END REDUNDANCY CHECK

	// var newPaths = _.reject([npath_over, npath_in], function(el, i, arr){
	// 	return _.isNull(el) || el.processed;
	// });

	// return newPaths;
	// var polarity = TracePathTool.readPolarity(trace_in);
	// var c = new paper.Path.Circle({
	// 	parent: trace_in.parent, 
	// 	position: intPt.clone(), 
	// 	name: "C"+ polarity +"T: terminal from fragmentation", 
	// 	radius: trace_in.style.strokeWidth, 
	// 	fillColor: pLetterToCLPolarity(polarity)
	// });

	// Graph.processTraceIntersections = function(){
// 	var scope = this;
// 	var conductiveTraces = ["CGP", "CVP", "CNP"];
// 	conductiveTracesA = BenderSVG.match(designer.circuit_layer.layer, { prefix: conductiveTraces });
// 	_.each(conductiveTracesA, function(el, i, arr){
// 		el.self = true;	
// 		conductiveTracesB = BenderSVG.match(designer.circuit_layer.layer, { prefix: conductiveTraces });
// 		conductiveTracesB = _.reject(conductiveTracesB, function(el2, i2, arr2){
// 			return el2.self || el2.processed;
// 		});
		
// 		intersects = TracePathTool.getAllIntersections(el, conductiveTracesB);
		
// 		el.processed = true;
// 		_.each(intersects, function(el2, i2, arr2){
// 			// console.log("SET", _.map(conductiveTracesB, function(el){
// 			// 	return el.id;
// 			// }));
			
// 			var pathIn = el2._curve.path;
// 			var pathOut = el2._curve2.path;
// 			var offsetA = pathIn.getOffsetOf(el2.point);

// 			var polarity = TracePathTool.readPolarity(el2._curve2.path);

// 			var c_int = new paper.Path.Circle({
// 					parent: pathIn, 
// 					name: "C"+ polarity +"T: terminal from fragmentation",
// 					position: pathIn.getPointAt(offsetA).clone() ,
// 					radius: pathIn.style.strokeWidth * 2, 
// 					fillColor: pathIn.style.strokeColor, 
// 					polarity: pathIn.polarity,
// 					strokeWidth: 1, 
// 					pathIds: [pathIn.id, pathOut.id] 
// 				});
// 			graph.addNode(c_int);
// 			// Graph.processIntersection(pathIn, pathOut, el2.point);
			
// 			// conductiveTracesB = BenderSVG.match(designer.circuit_layer.layer, { prefix: conductiveTraces });
// 			// conductiveTracesB = _.reject(conductiveTracesB, function(el3, i3, arr3){
// 				// return el3.self || el3.processed;
// 			// });
// 		});

// 		el.self = false;		
// 	});
// }


// 


// var polarity = TracePathTool.readPolarity(pathIn);
		// var c = new paper.Path.Circle({
		// 	parent: pathIn.parent, 
		// 	position: intPt.clone(), 
		// 	name: "C"+ polarity +"T: terminal from fragmentation", 
		// 	radius: pathIn.style.strokeWidth, 
		// 	fillColor: pathIn.style.fillColor,//pLetterToCLPolarity(polarity), 
		// 	processed: true
		// });
		// var npt = npath_in.getNearestPoint(intPt);
		// _.each([pathIn, npath_in], function(el, i, arr){
		// 	if(el.contains(intPt)) return;
		// 	if(el.id != 40) return;
		// 	var pos = el.getOffsetOf(intPt * 1.0) / el.length;
		// 	console.log("POS", pos, el.id);
		// 	if(pos > 0.5) el.addSegments([intPt.clone()])
		// 	else el.insertSegments(0, [intPt.clone()]);
		// // el.firstSegment.point.x += 10
			
		// // 		// el.firstSegment.point.x += 10
		// 	console.log(el, el.contains(intPt));
		// });
		// pathIn.addSegments([intPt.clone()]);
		// console.log(pathIn.id, pathOut.id, "made", npath_in.id);





		// _.each([pathOut, npath_over], function(el, i, arr){
		// 	if(el.contains(intPt)) return;
		// 	var pos = el.getOffsetOf(intPt * 1.0) / el.length;
		// 	if(pos > 0.5) el.addSegments([intPt.clone()])
		// 	else el.insertSegments(0, [intPt.clone()]);
		// });
		// console.log(pathIn.id, pathOut.id, "made", npath_over.id);

		// console.log("CONTAINS OUT", "pathOut", pathOut.contains(intPt), "npath_over", npath_over.contains(intPt));

		// var npt = npath_over.getNearestPoint(intPt);
		// var npt_offset = pathOut.getOffsetOf(npt);
		// var midPath = pathOut.length / 2.0;
		// if(npt_offset > midPath){
		// 	pathOut.addSegments([intPt.clone()]);
		// 	console.log("Made end", npt_offset, pathOut.length, npath_over.id);
		// }
		// else{
		//    pathOut.insertSegments(0, [intPt.clone()]);	
		//    // console.log("Made start", npt_offset, pathOut.length,  npath_over.id);		
		// }




Graph.processIntersection =  function(pathIn, pathOut, intPt){
	var offsetA = pathIn.getOffsetOf(intPt);
	var offsetB = pathOut.getOffsetOf(intPt);
	
	var npath_in = null;
	var npath_over = null;

	if(!_.isNull(offsetA) && offsetA > 2 && offsetA < (pathIn.length - 2) ){
		npath_in = pathIn.split(offsetA);
		
		npath_in.processed = true;
	}
	if(!_.isNull(offsetB) && offsetB > 2 && offsetB < (pathOut.length - 2)){
		npath_over = pathOut.split(offsetB);

		
	}
	

}

this.mode = TransformTool2.TOUCH;
	if(this.mode != TransformTool2.TOUCH){
		this.tool.onMouseDown = function(event){
			hitResult = scope.paper.project.hitTest(event.point, hitOptions);
			
			if(_.isNull(hitResult)) scope.canvas_item_type = "canvas";
			else{
				path = hitResult.item;
				if(path.name == "selection rectangle") scope.canvas_item_type = "transform";
				else scope.canvas_item_type = "pan";
			} 
			console.log("MouseDown", scope.canvas_item_type);
			scope[scope.canvas_item_type].onMouseDown(event, hitResult, scope);
			scope.update();
		}
		// rerouting


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

	}
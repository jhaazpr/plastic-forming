

Validator.RESISTANCE_THRESHOLD = 3000;
Validator.FAB_THRESHOLD = 250;
function Validator(graph){
	this.graph = graph;
}

Validator.prototype = {
	bom: function(){
		var leds = BenderSVG.get({name:"CP:_circuit_x5F_led_1_"});
		this.diodes = _.map(leds, function(led){
			return new Diode(led.id);
		});
		var total_path_length = _.reduce(solutions, function(memo, el){
			return memo + el.length;
		}, 0);
		total_path_length = Ruler.pts2mm(total_path_length);
		return [
					{level: 3, 
						elements: [], 
						message: "<b>Total path length</b>: "+ total_path_length.toFixed(0)+ " mm"
					}, 
					{level: 3, 
						elements: [], 
						message: "<b># of LEDs</b>: " + leds.length
					}, 
				]
	},
	validate: function(){
		graph.enable();
		bom = this.bom();
        msgsC = this.checkIfBatteryExists();
        msgsA = this.validateLEDsConnectedToGround();
        msgsB = this.validateOhmThreshold();
       // console.log(errorsA, errorsB);
        // return _.flatten([errorsC, msgsA, errorsB, ]);
        msgs =  _.flatten([bom, msgsC, msgsA, msgsB]);
      
        this.updateSidePanel(msgs);
        graph.disable();
	},
	checkIfBatteryExists: function(){
		r = graph.getSourceNode();
   		s = graph.getSinkNode();
   		console.log("BATTERY EXISTS", _.isNull(s) || _.isNull(r), r, s)
 		if(_.isNull(s) || _.isNull(r)) 
			return [{level: 5, 
						elements: [], 
						message: "You are missing a power source. Try adding a battery."}];
		else
			return [{level: 6, 
						elements: [], 
						glyph: "check",
						message: "Circuit is powered."}
						];
	},
	validateLEDsConnectedToGround: function(){
		var scope = this;
		console.log("Checking LEDS: " + this.diodes.length)
		

		console.log("Checking Power");
		var powered = _.map(this.diodes, function(diode, i){
			return diode.getPathsToPower();
		});


		console.log("Checking Grounded");
		var grounded = _.map(this.diodes, function(diode, i){
			return diode.getPathsToGround();
		});
	
		
		var errors = _.map(this.diodes, function(diode, i){
			var isGrounded = grounded[i].length > 0;
			var isPowered = powered[i].length > 0;
			console.log(diode.id, "POWERED?", isPowered, "GROUNDED?", isGrounded);
			if(!isGrounded && !isPowered){
				return {level: 5, 
							elements: [diode.id], 
							message: "The LED is not connected to ground or power."
						}

			}

			if(! isGrounded){
				return {level: 5, 
							elements: [diode.id], 
							message: "The LED is not connected to ground."
						}
			}
			if(! isPowered){
				return {level: 5, 
							elements: [diode.id], 
							message: "The LED is not connected to power."
						}
			}

			return false;
		});
		errors = _.compact(errors);
		if(errors.length == 0){
			return [{level: 6, 
							glyph: "check",
							elements: [], 
							message: "All LEDs connected to battery ground."
						}]
		}else{
			return errors;
		}
		return [];
	}, 
	validateOhmThreshold: function(){
		var scope = this;
		console.log("Checking Ohms: " + this.diodes.length)
		
		var powered = _.map(this.diodes, function(diode, i){
			return diode.getPathsToPower();
		});

		var errors = _.map(this.diodes, function(diode, i){
			var isPowered = powered[i].length > 0;
			
			if(isPowered){
				var ep = powered[i][0]; // shortest path to power
				ohmage = sr_model.apply(ep.solution);
				// console.log(ohmage);
				if(ohmage > Validator.RESISTANCE_THRESHOLD){
					return {level: 5,
									elements: [ep.solution.id], 
									message: "You have too much resistance (" + ohmage.toFixed(0) + "Ω) from this line (max= "+ Validator.RESISTANCE_THRESHOLD +"Ω). Try making a shorter path to the red battery terminal."
								}
				}
				
				if(ohmage > Validator.FAB_THRESHOLD){
					return {level: 5,
									elements: [ep.solution.id], 
									message: "You'll have trouble making this trace. Try making a shorter path to the red battery terminal."
								}
				}
				
			}
			return false;
		});
		errors = _.compact(errors);

		if(errors.length == 0){
			return [{level: 6, 
							glyph: "check",
							elements: [], 
							message: "Resistance looks good!"
						}]
		} else{
			return _.compact(errors);
		}
	}, 
	updateSidePanel: function(msgs){
		var scope = this;
		var dom = $('#debug-guide .list-group');
		dom.html("");	
		
		// BILL OF MATERIALS
		_.each(msgs, function(el, i, arr){
		   	guideDOM = Validator.generateSidePanelNode(el);
		   	dom.append(guideDOM);
		});
		
	}, 
}


Validator.generateSidePanelNode =  function(el){

				if(el.level == 1) level = "active";
				if(el.level == 2) level = "list-group-item";
				if(el.level == 3) level = "list-group-item-warning";
				if(el.level == 4) level = "list-group-item-info";
				if(el.level == 5) level = "list-group-item-danger";
				if(el.level == 6) level = "list-group-item-success";
		    	

		    	var guide_dom = $('<li class="list-group-item '+ level +'"></li>');
		    	var row = $('<div></div>').addClass("row");
		    	var col_a = $('<div></div>').addClass('col-xs-2');
		    	var col_b = $('<div></div>').addClass('col-xs-10');
		    	if(level == "active"){
		    		col_a = col_b;
		    		row.append(col_a);
		    		col_a.removeClass('col-xs-10').addClass('col-xs-12');
		    	}else
			    	row.append([col_a, col_b]);
		    	guide_dom.append(row);

		    	if(el.icon){
			    	var icon = "icon-" + el.icon;
			    	icon = $("<span class='"+ icon +" guide-icon'></span>");
			    	if(el.icon == "clock"){
			    		icon.addClass('timer').attr('id', "timer");
			    		icon.addClass('timer').attr('data-time', el.time);
			    	}

			    	col_a.prepend(icon);
			    }
			    if(el.glyph){
			    	var glyph = "glyphicon glyphicon-" + el.glyph;
			    	col_a.prepend("<span class='"+ glyph +" guide-icon'></span>");
			    }
			    if(el.debug){
			    	var btnGroup = DOM.tag("div").addClass('btn-group');
			    	var yesButton = DOM.tag("button").addClass('btn btn-sm btn-success').appendTo(btnGroup);
			    	var yes = DOM.tag("span").addClass('glyphicon glyphicon-ok').appendTo(yesButton);
			    	var noButton = DOM.tag("button").addClass('btn btn-sm btn-info').appendTo(btnGroup);
			    	var no = DOM.tag("span").addClass('sm-icon-debug').appendTo(noButton);
			    	var check = DOM.tag("span").addClass('glyphicon glyphicon-ok');
		    		

			    	var checkBehavior = function(){
			    		$(this).closest('.row').children('.col-xs-10').append(btnGroup);
			    		$(this).closest(".list-group-item").addClass('list-group-item-warning');
		    			$(this).remove();
		    			yesButton.click(yesBehavior);
				    	noButton.click(noBehavior);
			    	}
			    	var yesBehavior = function(){
			    		$(this).closest(".list-group-item").removeClass('list-group-item-warning');
			    		
				    	check.click(checkBehavior);
			    		$(this).closest('.row').children('.col-xs-2').append(check);
			    		$(this).parent().remove();
			    	}
			    	var noBehavior = function(){
			    		var elems = _.map(el.debug, function(g){
			    			return Validator.generateSidePanelNode(g);
			    		});
			    		// console.log(el.debug);
			    		elems = _.each(elems, function(e){
			    			var yesButton = DOM.tag("button").addClass('btn btn-sm btn-success').appendTo(btnGroup);
			    			var yes = DOM.tag("span").addClass('glyphicon glyphicon-ok').appendTo(yesButton);
			    			yesButton.click(function(){
			    				$(this).parent().parent().parent().remove();
			    			});
			    			e.find('.col-xs-10').append(yesButton);
			    		});
			    		$(this).parent().parent().parent().parent().after(elems);
			    		$(this).unbind('click');
			    		$(this).remove();
			 
			    	
			    	}
			    	check.click(checkBehavior);
			    	yesButton.click(yesBehavior);
			    	noButton.click(noBehavior);
			    }

			     var msg = DOM.tag("p").html(el.message);

			    col_b.append([msg, btnGroup]); 
			   
		    	if(el.multimeter){	
			    	guide_dom.attr('data-multimeter', el.multimeter);
			    	guide_dom.attr('data-probe-a', el.probeA);
			    	guide_dom.attr('data-probe-b', el.probeB);
		    		src = Multimeter.getSrc(el.multimeter);

	    			console.log(src);
    				var img = DOM.tag("img", true).attr('src', src).css({
    					width: "80%"
    				});
			    	msg.append(img);
				}		
				
			    // console.log("ELEMENT", el);
		    	guide_dom.attr('data-highlight', el.elements.join(','));
		        	

		    	guide_dom.hover(
		    		function(){
				    	// MULTIMETER LOGIC
		    			mm = $(this).attr('data-multimeter');
			    		if(! _.isUndefined(mm)){
				    		probeA = parseInt($(this).attr('data-probe-a'));
				    		probeB = parseInt($(this).attr('data-probe-b'));
				    		
			    			posA = graph.find(probeA).getPosition();
			    			posB = graph.find(probeB).getPosition();
			    			
			    		
			    			
			    			// console.log(src);
		    				// var img = DOM.tag("img", true).attr('src', src);
		    				// col_b.append(img);
					    }	
					    else multimeter.hide();

					    elements = $(this).attr('data-highlight').split(',');
					    elements = _.compact(elements);
					    // LIGHT UP THE ELEMENTS
					    _.each(elements, function(el, i, arr){
					    	var el = Node.get(parseInt(el));
					    	OhmTool.glow(el, true);
			    		});
			    	}, 
			    	function(){
			    		elements = $(this).attr('data-highlight').split(',');
			    		elements = _.compact(elements);
					   
			    		_.each(elements, function(el, i, arr){
			    			var el = Node.get(parseInt(el));
			    			OhmTool.glow(el, false);
			    		});
			    	}
			    );


				
		    	return guide_dom;
    }
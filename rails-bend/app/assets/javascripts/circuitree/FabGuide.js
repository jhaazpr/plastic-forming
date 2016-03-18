
function FabGuide(graph){
	this.dom = $('#fab-guide .list-group');
	var leds = BenderSVG.get({name:"CP:_circuit_x5F_led_1_"});
		this.diodes = _.map(leds, function(led){
			return new Diode(led.id);
		});

	// this.ptgs = graph.getPathsToGround();
	// this.ptgs = _.sortBy(this.ptgs, function(ptg){ return ptg.length;});
	// this.ptgs = _.uniq(this.ptgs, function(ptg){ 
	// 	return (ptg.length / 30).toFixed(0);
	// });
	// this.ptgs = _.each(this.ptgs, function(ptg, i){ 
	// 	console.log(i, ptg.length)
	// });


	// this.paths = _.map(this.ptgs, function(path){
	// 	return _.map(path.str.split('-'), function(el){ return parseInt(el);});
	// });
	this.battery = paper.project.getItems({name: "CP:_Battery"})[0];

	// var leds = paper.project.getItems({name:"CP:_circuit_x5F_led_1_"});
	// this.ledIDs = _.map(leds, function(led){
	// 		return led.id;
	// });
	// this.leds = _.map(leds, function(el){
	// 	positive_terminal = BenderSVG.match(el, {prefix: ["CVT"]})[0].path;
	// 	if(!_.isUndefined(positive_terminal)) positive_terminal = positive_terminal.terminals;
	// 	else positive_terminal = [];

	// 	negative_terminal = BenderSVG.match(el, {prefix: ["CGT"]})[0].path; 
	// 	if(!_.isUndefined(negative_terminal)) negative_terminal = negative_terminal.terminals;
	// 	else negative_terminal = [];

	// 	return _.flatten([positive_terminal, negative_terminal]); 
	// });
	// console.log("LED Termins", this.leds);
	// this.match = this.matchLEDsToPaths();

	// console.log("MATCH", match);

}

FabGuide.prototype = {
	fabricate: function(){
		graph.enable();
        this.updateSidePanel();
        graph.disable();
	},
	
	updateSidePanel: function(){
		var scope = this;
		this.dom.html("");	

		var powered = _.map(this.diodes, function(diode, i){
			return diode.getPathsToPower();
		});

		var grounded = _.map(this.diodes, function(diode, i){
			return diode.getPathsToGround();
		});

		var all_paths = _.map(this.diodes, function(diode, i){
			var all_paths = [diode.getAllPathsToFromPowerPad(), diode.getAllPathsToFromGroundPad()];
			all_paths = _.flatten(all_paths);
			return _.map(all_paths, function(el){ return el.mini_solution.id });
			// return [];
		});
		console.log("ALL PATHS", all_paths);

		var ptgs = _.map(this.diodes, function(diode, i){
			powered_path = powered[i][0];
			grounded_path = grounded[i][0];
			return {length: powered_path.length + grounded_path.length, powered: powered_path, grounded: grounded_path, diode: diode, all_paths: all_paths[i]};
		});
		
		ptgs = _.sortBy(ptgs, function(el){ return el.length;})
		// console.log(ptgs);


		var guides = _.map(ptgs, function(el, i, arr){
			dryingtime = 60;
			p_ohmage = sr_model.apply(el.powered.solution);
			n_ohmage = sr_model.apply(el.grounded.solution);
			
			p_ohmage_upperbound = 2 * p_ohmage; 
			n_ohmage_upperbound = 2 * n_ohmage; 
			p_ohmage_upperbound = p_ohmage_upperbound.toFixed(0);
			n_ohmage_upperbound = n_ohmage_upperbound.toFixed(0);

			var sourceNode = Node.get(graph.getSourceNode()).sourceNode;
			var sinkNode = Node.get(graph.getSinkNode()).sourceNode;

			
					// 	led_terminals = scope.leds[i];
		// 	// console.log("HGO", scope.ledIDs, scope.leds, led_terminals);
			var guide = [];
			if(i == 0){
				alltracesoutofbattery = _.flatten(_.map(ptgs, function(temp){ return [temp.powered.mini_solution.id, temp.grounded.mini_solution.id]}));
				guide.push(
					[{level: 1, 
						icon: "",
						elements: [], 
						message: "<b>STEP #" + 1 + ": ATTACH BATTERY </b>"},
					{level: 2, 
						icon: "pen",
						elements: alltracesoutofbattery, 
						message: "Draw all the traces coming out of your battery with your silver ink pen."},
					{level: 2, 
						icon: "place",
						elements: [scope.battery.id], 
						message: "Add the battery."},
					{level: 3, 
						icon: "multimeter",
						multimeter: "voltage", 
						probeA: sourceNode, probeB: sinkNode, elements: alltracesoutofbattery, 
						message: "<b class='check'>Power Check</b>: For all paths, position and <em>match</em> your probes to the dots. Is the reading approximately <b>3.3 V</b>?  ",
						debug: [
							{level: 4,
								icon: "", 
								elements: [scope.battery.id], 
								message: "Press the battery down with more pressure."
							}
						]
					}
					
					]
				);
			}

			
			var timer_updatedfied = DOM.tag("span").addClass("timer-uf").html(dryingtime);
			var timer_html = timer_updatedfied.wrap("<div>").parent().html();

			// new Timer(seconds, updatefield);
			var drawable_paths = _.flatten([el.powered.solution.id, el.grounded.solution.id, el.all_paths]);
			guide.push(
				[
				{level: 1, 
					icon: "",
					elements: [], 
					message: "<b>STEP #" + (i+ 2) + ": DRAW BRANCH </b>"},
				{level: 2, 
					icon: "pen",
					elements: drawable_paths, 
					message: "Draw Trace #" + (i+ 1) + " with your silver ink pen."},
				{level: 2, 
					icon: "clock",
					elements: drawable_paths, 
					message: "Let these traces dry for <b>"+ timer_html +" seconds</b>.", 
					time: dryingtime },
				{level: 3,
					icon: "multimeter", 
					multimeter: "resistance", 
					probeA: el.powered.nodes[0], 
					probeB: el.diode.positive_terminal,  
					elements: [el.powered.solution.id], 
					message: "<b class='check'>Resistance Check</b>: Position and <em>match</em> your probes to the dots. Is the reading <b>less than " + p_ohmage_upperbound + " Ω</b>? ",
					debug: [
							{level: 4,
								icon: "eye", 
								elements: [el.powered.solution.id], 
								message: "Let it dry a little longer and check again."}
						]
					},
				
				{level: 3,
					icon: "multimeter", 
					multimeter: "resistance", 
					probeA: el.grounded.nodes[0], 
					probeB: el.diode.negative_terminal,  
					elements: [el.grounded.solution.id], 
					message: "<b class='check'>Resistance Check</b>: Position and <em>match</em> your probes to the dots. Is the reading <b>less than " + n_ohmage_upperbound + " Ω</b>? ",
					debug: [
							{level: 4,
								icon: "eye", 
								elements: [el.grounded.solution.id], 
								message: "Let it dry a little longer and check again."
							}
						]
					},
				{level: 2, 
					icon: "place",
					elements: [el.diode.id], 
					message: "Place the following LED. Pay attention to orientation."},
				{level: 3, 
					icon: "eye",
					elements: [el.diode.id], 
					message: "Do the LED(s) turn on?", 
				debug: [
						{level: 4,
							icon: "eye", 
							elements: [el.diode.id], 
							message: "Press the LED(s) down with more pressure."}
						],
				}
				
				]
			);
		timer_updatedfied.unwrap();
		// 	// LEAVE BATTERY ON IF ITS THE LAST ONE
		// 	// if(arr.length - 1 == i) guide = guide.slice(0, guide.length - 1);
			return _.flatten(guide);
		});
		guides = _.flatten(guides);
		this.setGuides(guides);	
	}, 
	setGuides: function(guides){
		var scope = this;
		_.each(guides, function(el, i, arr){
			guides = Validator.generateSidePanelNode(el);

			scope.dom.append(guides)
		});
		$('.timer').each(function(i, el){
			// console.log(i, el);
			$(el).click(function(e){
				var uf = $(this).parent().parent().find('.timer-uf');
				var t =  parseInt($(this).attr('data-time'));
				// console.log("Starting timer for", t, uf);
				
				var timer = new Timer(t, uf);
				timer.start();
			});
		});
	}
}
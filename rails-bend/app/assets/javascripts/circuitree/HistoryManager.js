// HistoryManager.js

function HistoryManager(storage){
	this.storage = storage;
	this.current_save = this.getHead();
	// If there is no save history
	console.log("HISTORY ITEMS" + this.getHistory().length);
	if(this.getHead() == Number.NEGATIVE_INFINITY){
		console.log("No history in cache");
		resp = $.getJSON(design.json.url, function(resp){
       		designer.loadJSON(resp);
    	});
		this.save();
	}else{
		this.trimToHead();
		this.loadEvent(this.current_save);
	}
	// If there is, just keep the head
	this.current_save = this.getHead();	
}


HistoryManager.prototype = {
	clear_all: function(){
		this.clear_history();
		this.current_save = this.getHead();
		console.log("clear all", this.current_save)
		var scope = this;
		resp = $.getJSON("/blank.json", function(resp){
       		designer.loadJSON(resp, CircuitDesigner.BLANK_CANVAS);
  //   		scope.server_save();
			// scope.save();
			scope.current_save = scope.getHead();
			zoom.home();
    	});
	},
	loadEvent: function(t){
		designer.loadJSON(eval(storage.get('saveevent_' + t)), CircuitDesigner.BLANK_CANVAS);
	},
	trimToHead: function(){
		var save_events = this.getHistory();

		var scope = this;
		var rel_events = _.filter(save_events, function(t){
			return t > scope.current_save; 
		});
		_.each(rel_events, function(t){
			storage.remove("saveevent_" + t);
		});
		// console.log("Pruned the tree by", rel_events.length, "elements.");
	},
	getHistory: function(){
		save_events = $.map(storage.keys(), function(el, i){
			flag = el.split('_')[0];
			time = parseInt(el.split('_')[1]);
			if(flag == "saveevent")
				return time;
		});	
		return save_events;
	}, 
	getHead: function(){
		return _.max(this.getHistory());
	}, 
	server_save: function(){
		paper.view.update();
		var json  =  designer.json();
		
		var name = design.name.trim();
		var data = {
			json: json, 
			name: name
		}
		console.log(json);
		$.ajax({
		  url: '/designs/' + design.id + "/design_update",
		  type: 'POST',
		  data: data,
		  success: function(data) {
		    console.log(data);
		  }
		});

		// sys.log("Saving to the server!");
	},
	save: function(){
		designer.clearForSave();
		
		var s = Math.floor(Date.now() / 1000);
		var key = "saveevent_" + s;
	
		// PRUNE THE TREE
		if(this.current_save != this.getHead()){
			// If you are not at head of the list, then remove. 
			this.trimToHead();
		} 
	
		storage.set(key, designer.json());
		this.current_save = s;
		
		designer.unclearForSave();
	}, 
	redo: function(){
		designer.clearForSave();
		var save_events = this.getHistory();

		var scope = this;
		// GET ALL EVENTS THAT OCCURED AFTER
		var rel_events = _.filter(save_events, function(t){
			return t > scope.current_save; 
		});
		
		if(_.isEmpty(rel_events)){ 
			sys.show("Can't redo...");
			return;
		}
		rel_event = _.min(rel_events);

		console.log("Redoing", rel_event);
		
		this.loadEvent(rel_event);
		this.current_save = rel_event;
		sys.show("Redoing last action");
		designer.unclearForSave();
	},
	undo: function(){
		designer.clearForSave();
		var save_events = this.getHistory();
		var scope = this;
		var rel_events = _.filter(save_events, function(t){
			return t < scope.current_save; 
		});
		sys.log("rel_events:" + rel_events.join("<br/>"));
		if(_.isEmpty(rel_events)){ 
			sys.show("Can't undo...");
			return;
		}
		rel_event = _.max(rel_events);
		
		designer.loadJSON(eval(storage.get('saveevent_' + rel_event)), CircuitDesigner.BLANK_CANVAS);
		
		this.current_save = rel_event;
		sys.show("Loading:" + rel_event);
		designer.unclearForSave();
	}, 
	clear_history: function(){
		storage.clear();
		sys.show("Clearing browser history.");
	}
}


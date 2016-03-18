function Timer(seconds, updatefield){
  // console.log("Creating a timer", seconds, updatefield);
  this.time_remaining = seconds;
  this.dom = updatefield;
}
Timer.prototype = {
  start: function(){
    this.dom.html(this.time_remaining);
    var self = this;
    var timeinterval = setInterval(function(){
      self.time_remaining --;
      self.dom.html(self.time_remaining);
      if(self.time_remaining <= 0){
        clearInterval(timeinterval);
        beep();
        for(i=0;i<3;i++) {
          self.dom.parent().fadeTo('slow', 0.5).fadeTo('slow', 1.0);
        } 
      }
    }, 1000);
  }
}
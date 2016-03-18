function PaperSetup(){

}
PaperSetup.types = {
	A4: { 
		  width: 210, 
		  height: 297
		}
}
PaperSetup.orientation = function(paper_type, t){
	if(t == "hoz"){
		return {width: paper_type.height, height: paper_type.width}
	}
	else return paper_type;
}
class JigController < ApplicationController
  def designer

  	render :layout => "full_screen"
  end

  def generator
    render :layout => "full_screen"
  end
  def form
    design = Design.find(2)
    @guides = design.fab_guides.to_json.html_safe
  	@files = get_primitives()
  	render :layout => "full_screen"
  end
  def interface
    design = Design.find(params[:id])
    @design = design.to_json.html_safe
    @design_obj = design
    @files = get_primitives()
    # @guides = design.fab_guides.to_json.html_safe
    render :layout => "full_screen"
    # render :json =>  
  	# render :json => @files
  end

  def bom
  end


  def upload
    uploaded_io = params["file"]
    File.open(Rails.root.join('public', 'primitives', "uploads_" + uploaded_io.original_filename), 'wb') do |file|
      file.write(uploaded_io.read)
    end
    render :json => "uploads_" + uploaded_io.original_filename
  end
  def primitives
    @files = get_primitives()
    render :json => @files
  end

  # HELPER METHODS
  def get_primitives
    files = {path: "/primitives/", filenames: Dir.glob("public/primitives/*").collect!{|c| c.split('/')[2..-1].join('/')}}
    files[:filenames].collect!{|f| {:collection => f.split('.')[0].split('-')[0].split('_')[0].titlecase, :filename => f, :title => f.split(".")[0].titlecase}}
    files
  end
end

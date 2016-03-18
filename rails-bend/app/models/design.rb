class Design < ActiveRecord::Base
	mount_uploader :json, JsonUploader
	has_many :fab_guides
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LABEL_GENERATOR - For creating image labels by masking superpixels

% Asks the user to select a folder containing images, the loops through
% each image in the folder to process it. Superpixels will be formed from
% the image and displayed, and when a superpixel is clicked on it will be
% added to the mask.
%
% A new folder will be created with '_label' appending to it, and the
% generated masks will be saved there. Any image that already has a
% corresponding mask in the output folder will be skipped.
%
% The mex file for the SLIC superpixel algorithm must be generated 
% prior to use by running "mex slicmex.c"
%
% Alternatively, the SLICO algorithm can be used, but this tends to
% produce too uniform of superpixels, and is not good for extracting
% complex shapes.
%
% Controls:
%   -Click: Add/subract superpixel to/from mask
%   -Up Arrow: Increase number of superpixels
%   -Down Arrow: Decrease number of superpixels
%   -R: Reset image
%   -S: Save image
%   -Space: Next image (without saving)
%   -ESC: Terminate
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters
img_type = '.jpg';
num_superpixels_initial = 25;

% Get directory to desired folder
img_folder_path = uigetdir([pwd, '/..']);
[directory, img_folder_name] = fileparts(img_folder_path);

% Form the output folder name and directory
label_folder_name = [img_folder_name, '_labels'];
label_folder_path = fullfile(directory, label_folder_name);
if ~exist(label_folder_path, 'dir')
    % Create if needed
    mkdir(directory, label_folder_name);
end

% Get a list of all the images
files = fullfile(img_folder_path, ['*', img_type]);
img_list = dir(files);
num_images = size(img_list, 1);

% Loop through images
for i = 1:num_images
    % From full names for input and output images
    [~, img_name_no_ext] = fileparts(img_list(i).name);
    img_name_full = fullfile(img_folder_path, [img_name_no_ext, img_type]);
    label_name_full = fullfile(label_folder_path, [img_name_no_ext, '_label', img_type]);
    
    % Only process if a label has not been created yet
    if ~exist(label_name_full, 'file')
        
        % Load the image
        img = imread(img_name_full);
        img_OG = img;
        
        set(gcf,'name', img_name_no_ext, 'numbertitle', 'off')
        im_w = size(img, 2);
        im_h = size(img, 1);
        generate_superpixels = true;
        mask = false(im_h, im_w);
        num_superpixels = num_superpixels_initial;
        while (true)
            
            if (generate_superpixels)
                % Run Slic algorithm
                [labels, numlabels] = slicmex(img, num_superpixels, 15);
                
                %Create perimeters by looping through clusters and inspecting pixels
                perim_mask = true(im_h, im_w);
                for k = 1:(numlabels - 1)
                    regionK = labels == k;
                    perimK = bwperim(regionK, 8);
                    perim_mask(perimK) = false;
                end
                
                %Overlay perimeters on image
                perim_mask = uint8(cat(3,perim_mask, perim_mask, perim_mask));
                %perim_mask = uint8(perim_mask);
                img_superpixels = img.*perim_mask;
                
                warning('off', 'all')
                imshow(img_superpixels)
                warning('on', 'all')
            end
            
            % Get the click point
            [xi, yi, but] = ginput(1);
            switch but
                case 27
                    % ESC key pressed
                    f = gcf;
                    close(f);
                    return;
                case 1
                    % Get which superpixel number was picked, and the current state
                    % of the mask at that point
                    label_num = labels(round(yi), round(xi));
                    state = mask(round(yi), round(xi));
                    
                    % From mask
                    mask(labels == label_num) = ~state;
                    g_mask = cat(3, false(size(mask)), mask, false(size(mask)));
                    rb_mask = cat(3, mask, false(size(mask)), mask);
                    
                    % Apply mask to the image
                    img = img_OG;
                    img(g_mask) = img(g_mask)*1.4;
                    img(rb_mask) = img(rb_mask)*0.6;
                    
                    warning('off', 'all')
                    imshow(img.*perim_mask)
                    warning('on', 'all')
                    
                    generate_superpixels = false;
                case 30
                    % Up button pressed
                    % Increase number of superpixels
                    num_superpixels = num_superpixels + 50;
                    generate_superpixels = true;
                case 31
                    % Down button pressed
                    % Decrease number of superpixels
                    num_superpixels = num_superpixels - 50;
                    generate_superpixels = true;
                case 114
                    % R button pressed
                    % Reset the image and mask
                    mask = false(im_h, im_w);
                    img = img_OG;
                    warning('off', 'all')
                    imshow(img.*perim_mask)
                    warning('on', 'all')
                case 115
                    % S button pressed
                    % Save the image
                    imwrite(mask, label_name_full)
                    break;
                case 32
                    % Space bar pressed
                    % Skip to next image
                    break;
                otherwise
                    generate_superpixels = false;
            end
        end
    end
end

% Clean up
f = gcf;
close(f);

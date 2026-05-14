function [dest_path, d_n] = create_dir(dir_path, dir_name)
%% create_dir create a folder in the directory dir_path with 
% directory name:  dir_name [yyyy-mm-dd] 
% If directory already exists then append _1, _2, ...
%%

folder_id = 0;
done = false;
while ~done
    if folder_id ~= 0
        d_n = [dir_name '_' num2str(folder_id)];
    else
        d_n = dir_name;
    end
    dest_path = fullfile(dir_path, d_n);

    [~, msg] = mkdir(dest_path);
    
    done = ~strcmp(msg, 'Directory already exists.');
    folder_id = folder_id + 1;
end

end


%% Create DTI matrix
fileNames = dir('KinaseMATfiles/');

% First pass - collect all possible coumpound PubChem IDs - Maintain compound
% dictionary
a = [];
for fn = 1:length(fileNames)-2
    fname = fileNames(fn+2).name
    load(strcat('KinaseMATfiles/',fname));
    a = [a; RAW(2:end,3)];
end

out=cellfun(@num2str,a,'un',0);
[uni, ia, ic] = unique(out);



% Visualize frequency of each compound
t = zeros(size(uni)); 
for i = 1:length(uni)
    t(i) = sum(ic == i);
end

%%%%%%
% We have 726 unique compound IDs, 246 kinases
dti = zeros(726, 246);
kinases = {};

for fn = 1:length(fileNames)-2
    fname = fileNames(fn+2).name
    load(strcat('KinaseMATfiles/',fname));
    kinases{fn} = fname(1:end-4);
    a = RAW(2:end,3);
    conc = RAW(2:end,7);
    for i = 1:length(a)
        ind = find(strcmp(uni, num2str(a{i})));
        temp = conc(i);
        if temp{1} < 1000
            dti(ind,fn) = 1;
        end
    end
end
%Delete the N/A row - lots of compounds don't have PubChem IDs
dti = dti(1:725,:);



%Store Pubchem IDs
fid = fopen('PubChemIDs.csv', 'w') ;
fprintf(fid, 'PID\n') ;
fprintf(fid, '%s\n', uni{1:end-1,1}) ;
fclose(fid) ;
%dlmwrite('PubChemIDs.csv', uni(2:end,:), '-append') ;
%Store Kinase names
kinases = reshape(kinases, [246,1]);
fid = fopen('Kinases.csv', 'w') ;
fprintf(fid, 'KID\n') ;
fprintf(fid, '%s\n', kinases{1:end,1}) ;
fclose(fid) ;

%Store DTI table
T= table(dti);
writetable(T,'dti2.csv');





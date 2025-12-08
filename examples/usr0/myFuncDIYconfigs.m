function configs = myFuncDIYconfigs(configs)

% === processing mooring from Cases ==================================

    itypeA = 1;
    itypeS = 2;
 
    thisX2F = configs.lineTypes.lineType(itypeA).X2F;
    thisZ2F = configs.lineTypes.lineType(itypeA).Z2F;
    % attribute line types
    configs.lineSys.anchorLineType = ones(configs.lineSys.nAnchorLine)*itypeA;
    configs.lineSys.sharedLineType = ones(configs.lineSys.nSharedLine)*itypeS;
    fprintf("Linetypes are distributed\n")
    
    dis1 = sqrt(thisX2F^2-500^2);
    dis2 = 500;
    configs.lineSys.anchorPos = [[thisX2F*cosd(240), thisX2F*sind(240), thisZ2F] + configs.lineSys.fairleadPos_init(:,4)';...
                                 [dis2, -dis1, thisZ2F]                          + configs.lineSys.fairleadPos_init(:,4)';...
                                 [thisX2F*cosd(300), thisX2F*sind(300), thisZ2F] + configs.lineSys.fairleadPos_init(:,8)';...
                                 [-dis1, dis2, thisZ2F]                          + configs.lineSys.fairleadPos_init(:,3)';...
                                 [ dis1, dis2, thisZ2F]                          + configs.lineSys.fairleadPos_init(:,5)';...
                                 [-dis1, dis2, thisZ2F]                          + configs.lineSys.fairleadPos_init(:,11)';...
                                 [ dis1, dis2, thisZ2F]                          + configs.lineSys.fairleadPos_init(:,13)';...
                                 [thisX2F*cosd(120), thisX2F*sind(120), thisZ2F] + configs.lineSys.fairleadPos_init(:,18)';...
                                 [dis2,  dis1, thisZ2F]                          + configs.lineSys.fairleadPos_init(:,18)';...
                                 [thisX2F*cosd(60),  thisX2F*sind(60),  thisZ2F] + configs.lineSys.fairleadPos_init(:,22)';...
                                 ]';

   fprintf("Anchor position is inited\n")

% for ibod = 1:length(configs.lineSys.floatBody)
%     configs.lineSys.floatBody(ibod).CClin(1,1) = 3.7e4;
%     configs.lineSys.floatBody(ibod).CClin(2,2) = 3.7e4;
% end


end
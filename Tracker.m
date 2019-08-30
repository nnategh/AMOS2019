classdef Tracker < handle
    % Classify spots to starts or objects then track objects
    
    % Properties
    properties (Constant)
        ASSETS_FOLDER = './assets';
        DELTA = [1, 5, 10, 15, 20, 25, 30];
        UNMATCHED_COST = [1, 2, 4, 8, 16, 32, 64];
    end
    
    properties
        folders % path of folders
        filenames % path of filenames
        binarizationThreshold % threshold of binarization
        unmatchedCost % cost of unmatched (unknown labels)
        maxNumObjects % maximum number of objects
        delta % length of memory (time-delta : time)
        isVelocityMagnitude % use magnitude of velocity to find outliers
        isVelocityDirection % use direction of velocity to find outliers
    end
    
    % Constructor
    methods
        function this = Tracker(...
                filename, ...
                binarizationThreshold, ...
                unmatchedCost, ...
                maxNumObjects, ...
                delta, ...
                isVelocityMagnitude, ...
                isVelocityDirection)
            % Constructor
            
            this.binarizationThreshold = binarizationThreshold;
            this.unmatchedCost = unmatchedCost;
            this.maxNumObjects = maxNumObjects;
            this.delta = delta;
            this.isVelocityMagnitude = isVelocityMagnitude;
            this.isVelocityDirection = isVelocityDirection;
            
            this.initFolders(filename);
            this.initFilenames(filename);
        end
        
        % Folders/Filenames
        function initFolders(this, filename)
            % Init `Folders` property
            
            this.folders = struct(...
                'assets', Tracker.ASSETS_FOLDER, ...
                'results', Tracker.getResultsFolder(filename));
        end
        
        function initFilenames(this, filename)
            % Init `Filenames` property
            
            [folder, name] = fileparts(filename);
            
            this.filenames = struct(...
                'video', filename, ...
                'data', fullfile(folder, [name, '.mat']), ...
                'output', fullfile(this.folders.results, sprintf('%s-output-d-%d-c-%d.mat', name, this.delta, this.unmatchedCost)), ...
                'classifiedVideo', fullfile(this.folders.results, sprintf('%s-classified-d-%d-c-%d.mp4', name, this.delta, this.unmatchedCost)));
        end
        
        % Save
        function save(this)
            props = {};
            
            props.filename = this.filenames.video;
            props.binarizationThreshold = this.binarizationThreshold;
            props.unmatchedCost = this.unmatchedCost;
            props.maxNumObjects = this.maxNumObjects;
            props.delta = this.delta;
            props.isVelocityMagnitude = this.isVelocityMagnitude;
            props.isVelocityDirection = this.isVelocityDirection;
            
            save(this.filenames.output, 'props');
        end
    end
    
    methods (Static)
        % Load
        function tracker = load(filename)
            load(filename, 'props');
            
             tracker = Tracker(...
                props.filename, ...
                props.binarizationThreshold, ...
                props.unmatchedCost, ...
                props.maxNumObjects, ...
                props.delta, ...
                props.isVelocityMagnitude, ...
                props.isVelocityDirection);
        end
    end
    
    % Classify
    methods
        function classify1(this)
            % Classify spots to
            %   - true: Object
            %   - false: Star
            %   - NaN: Unknown
            
            % properties
            th = this.binarizationThreshold; % binarization threshold
            uc = this.unmatchedCost; % unmatched/unknown cost
            mo = this.maxNumObjects; % maximum number of objects
            n = this.delta; % memory length
            
            memory = cell(n, 1); % memory
            centroids = cell(n, 1); % centroids
            labels = cell(n, 1); % labels
            matches = cell(n, 1); % matches
            translations = cell(n, 1); % translations
            
            % load sample video
            video = this.loadVideo();
            
            fprintf('\nCompute centroid labels: ...\n');
            tic();
            
            % read first frames to memory
            for t = 1:n
                 C1 = Tracker.getCentroids(readFrame(video), th); % centroids of previous frame
                 memory{t} = struct('t', t, 'C', C1);
                 centroids{t} = C1;
                 labels{t} = [];
                 matches{t} = struct('t', [], 'M', []);
                 translations{t} = [];
                 
                 fprintf('Frame: %d, \tSpots: %d, \tSkipped\n', t, size(C1, 1));
            end
            
            
            C1 = memory{1}.C;
            t2 = n + 1; % time of current frame
            while hasFrame(video)
                t1 = memory{1}.t; % time of previous frame
                
                C2 = Tracker.getCentroids(readFrame(video), th); % centroids of next frame
                
                % number of spots are same as previous frames
                n1 = size(C1, 1);
                n2 = size(C2, 1);
                
                txt = sprintf('Frame: %d, \tSpots: %d', t2, n2);
                
                if abs(n1 - n2) < mo
                    L = nan(n2, 1); % labels of current iteration
                    
                    % matching
                    M = matches{t1}.M;
                    if isempty(M)
                        D = Tracker.getCostMatrix(C1, C2);
                    else
                        T = zeros(size(C1));
                        T(M(:, 2), :) = translations{t1};
                        
                        D = Tracker.getCostMatrix(C1 + T, C2);
                    end
                    
                    
                    M = matchpairs(D, uc);

                    % translations
                    T = Tracker.getTranslations(C1, C2, M);

                    % labeling
                    TF = false(size(M, 1), 1);

                    if this.isVelocityMagnitude
                        TF = TF | ...
                            isoutlier(...
                                vecnorm(T, 2, 2), ...
                                'gesd', ...
                                'MaxNumOutliers', mo);
                            
%                         TF = TF | ...
%                             isoutlier(...
%                                 vecnorm(T, 2, 2));
                    end
                    if this.isVelocityDirection
                        TF = TF | ...
                            isoutlier(...
                                atan2(T(:, 2), T(:, 1)), ...
                                'gesd', ...
                                'MaxNumOutliers', mo);

%                         TF = TF | ...
%                             isoutlier(...
%                                 atan2(T(:, 2), T(:, 1)));
                    end

                    L(M(:, 2)) = TF; 

                    memory(1) = [];
                    memory{n} = struct('t', t2, 'C', C2);
                    C1 = memory{1}.C;
                    
                    txt = [txt, ...
                        sprintf('\tObjects: %d, \tUnknonw: %d\n', ...
                        sum(L == 1), sum(isnan(L)))];
                elseif n1 > n2
                    M = [];
                    t1 = [];
                    L = [];
                    T = [];
                    
                    memory(1) = [];
                    memory{n} = struct('t', t2, 'C', C2);
                    C1 = memory{1}.C;
                    
                    
                    
                    txt = [txt, '\tSkipped\n'];
                else
                    M = [];
                    t1 = [];
                    L = [];
                    T = [];
                    
                    txt = [txt, '\tSkipped\n'];
                end
                
                fprintf(txt);
                
                centroids{end + 1} = C2;
                labels{end + 1} = L;
                matches{end + 1} = struct('t', t1, 'M', M);
                translations{end + 1} = T;
                
                t2 = t2 + 1;
            end
            
            fprintf('\nFile `%s` saved: ', this.filenames.output);
            this.save();
            save(this.filenames.output, 'centroids', 'labels', 'matches', 'translations', '-append');
            toc();
        end
        
        function classify(this)
            % Classify spots to
            %   - true: Object
            %   - false: Star
            %   - NaN: Unknown
            
            % properties
            uc = this.unmatchedCost; % unmatched/unknown cost
            K = this.maxNumObjects; % maximum number of objects
            d = this.delta; % memory length
            
            memory = cell(d, 1); % memory
            centroids = cell(d, 1); % centroids
            labels = cell(d, 1); % labels
            matches = cell(d, 1); % matches
            translations = cell(d, 1); % translations
            
            % load sample video
            dr = DataReader(this.filenames.data);
            N = dr.N; % number of frames
            C = dr.pp.spot; % pixel positions of spots
            
            % fprintf('\nCompute centroid labels: ...\n');
            % tic();
            
            % read first frames to memory
            for i = 1:d
                 C1 = C{i}; % centroids of previous frame
                 memory{i} = struct('t', i, 'C', C1);
                 centroids{i} = C1;
                 labels{i} = [];
                 matches{i} = struct('t', [], 'D', [], 'M', []);
                 translations{i} = [];
                 
                 % fprintf('Frame: %d, \tSpots: %d, \tSkipped\n', i, size(C1, 1));
            end
            
            
            C1 = memory{1}.C;
            t2 = d + 1; % time of current frame
            for i = d + 1:N
                t1 = memory{1}.t; % time of previous frame
                
                C2 = C{i};
                
                % number of spots are same as previous frames
                n1 = size(C1, 1);
                n2 = size(C2, 1);
                
                txt = sprintf('Frame: %d, \tSpots: %d', t2, n2);
                
                if abs(n1 - n2) < K
                    L = nan(n2, 1); % labels of current iteration
                    
                    % matching
                    M = matches{t1}.M;
                    
                    if isempty(M)
                        D = Tracker.getCostMatrix(C1, C2);
                    else
                        T = zeros(size(C1));
                        T(M(:, 2), :) = translations{t1};
                        
                        D = Tracker.getCostMatrix(C1 + T, C2);
                    end
                    
                    
                    M = matchpairs(D, uc);
                    nm = size(M, 1); % number of match pairs

                    % translations
                    T = Tracker.getTranslations(C1, C2, M);

                    % labeling
                    TF = false(nm, 1);
                    

                    if this.isVelocityMagnitude && nm > 0
                        TF = TF | ...
                            isoutlier(...
                                vecnorm(T, 2, 2), ...
                                'gesd', ...
                                'MaxNumOutliers', min(K, nm));
                            
%                         TF = TF | ...
%                             isoutlier(...
%                                 vecnorm(T, 2, 2));
                    end
                    if this.isVelocityDirection && nm > 0
                        TF = TF | ...
                            isoutlier(...
                                atan2(T(:, 2), T(:, 1)), ...
                                'gesd', ...
                                'MaxNumOutliers', min(K, nm));

%                         TF = TF | ...
%                             isoutlier(...
%                                 atan2(T(:, 2), T(:, 1)));
                    end

                    L(M(:, 2)) = TF; 

                    memory(1) = [];
                    memory{d} = struct('t', t2, 'C', C2);
                    C1 = memory{1}.C;
                    
                    txt = [txt, ...
                        sprintf('\tObjects: %d, \tUnknonw: %d\n', ...
                        sum(L == 1), sum(isnan(L)))];
                elseif n1 > n2
                    D = [];
                    M = [];
                    t1 = [];
                    L = [];
                    T = [];
                    
                    memory(1) = [];
                    memory{d} = struct('t', t2, 'C', C2);
                    C1 = memory{1}.C;
                    
                    
                    
                    txt = [txt, '\tSkipped\n'];
                else
                    D = [];
                    M = [];
                    t1 = [];
                    L = [];
                    T = [];
                    
                    txt = [txt, '\tSkipped\n'];
                end
                
                % fprintf(txt);
                
                centroids{end + 1} = C2;
                labels{end + 1} = L;
                matches{end + 1} = struct('t', t1, 'D', D, 'M', M);
                translations{end + 1} = T;
                
                t2 = t2 + 1;
            end
            
            % fprintf('\nFile `%s` saved: ', this.filenames.output);
            this.save();
            save(this.filenames.output, 'centroids', 'labels', 'matches', 'translations', '-append');
            % toc();
        end
        
        function video = loadVideo(this)
            fprintf('Load `%s`: ', this.filenames.video);
            tic();
            video = VideoReader(this.filenames.video);
            toc();
        end
    end
    
    methods (Static)
        function C = getCentroids(I, T)
            BW = imbinarize(rgb2gray(I), T); % binary image

            C = regionprops(BW, 'Centroid');
            C = [C.Centroid];
            C = reshape(C, 2, [])';
        end
        
        function D = getCostMatrix(C1, C2)
            n1 = size(C1, 1);
            n2 = size(C2, 1);
            D = zeros(n1, n2);
            for i = 1:n1
              D(i,:) = vecnorm([C1(i, 1) - C2(:, 1), C1(i, 2) - C2(:, 2)], 2, 2)';
            end
        end
        
        function TC = getTotalCost(D, M)
            % assigned cost
            AC = sum(D(sub2ind(size(D), M(:,1), M(:,2))));
            % unassigned cost
            UC = costUnmatched * (sum(size(Cost)) - 2 * size(M,1));
            % total cost
            TC = AC + UC;
        end
        
        function T = getTranslations(C1, C2, M)
            % Get tranlation vectors
            T = C2(M(:, 2), :) - C1(M(:, 1), :); % [dx, dy]
        end
    end
    
    % Classification Performance
    methods
        function cp = getClassificationPerformance(this)
            % - ul: 0 | 1
            %   Unknown label
            
            load(this.filenames.output, 'labels');
            d = this.delta;
            
            dr = DataReader(this.filenames.data);
            N = dr.N;
            
            TPR = nan(N, 1);
            TNR = nan(N, 1);
            UNK = nan(N, 1);
            
            for i = (d + 1):N
                y_ = labels{i};
                n = numel(y_);
                
                if n == 0
                    continue;
                end
                
                y = zeros(n, 1);
                y(dr.idx.obj{i}) = 1;
                
                P = sum(y == 1);
                if P
                    TPR(i) = sum(y_ == 1 & y == 1) / sum(y == 1);
                else
                    TPR(i) = 1;
                end
                
                TNR(i) = sum(y_ == 0 & y == 0) / sum(y == 0);
                UNK(i) = sum(isnan(y_))/ numel(y_);
            end
            
            cp = struct(...
                'TPR', struct('mean', nanmean(TPR), 'std', nanstd(TPR)), ...
                'TNR', struct('mean', nanmean(TNR), 'std', nanstd(TNR)), ...
                'UNK', struct('mean', nanmean(UNK), 'std', nanstd(UNK)));
        end
        
        function plotClassificationAccuracy(this, ul)
            cp = this.getClassificationPerformance(ul);
            edgeColor = 'none';
            faceAlpha = 0.9;
            
            DataReader.createFigure('Classification - Number of Unknonws');
            % figure('Name', 'Classification - Number of Unknonws', 'Color', [1, 1, 1]);
            area(cp.ACC, ...
                'DisplayName', 'Target Objects', ...
                'EdgeColor', edgeColor, ...
                'FaceAlpha', faceAlpha);
            
            if ul
                title('Unknonws are Objects');
            else
                title('Unknonws are Stars');
            end
                
            xlabel('Frame')
            ylabel('Accuracy');
            
            nf = numel(cp.ACC); % number of frames
            xticks([1, this.delta, nf]);

            ylim([0.95, 1]);
            yticks([0.95, 0.99, 1]);
            box('off');
            this.setFontSize();
            
            this.saveFigure(sprintf('classification-accuracy-ul-%d', ul));
        end
        
        function plotUnknonwNumbers(this)
            
            cp = this.getClassificationPerformance();
            edgeColor = 'none';
            faceAlpha = 0.9;
            
            DataReader.createFigure('Classification - Number of Unknonws');
            % figure('Name', 'Classification - Number of Unknonws', 'Color', [1, 1, 1]);
            area(cp.NO, ...
                'DisplayName', 'Target Objects', ...
                'EdgeColor', edgeColor, ...
                'FaceAlpha', faceAlpha);
            hold('on');
            area(cp.NO_, ...
                'DisplayName', 'Output Objects', ...
                'EdgeColor', edgeColor, ...
                'FaceAlpha', faceAlpha);
            area(cp.NU, ...
                'DisplayName', 'Output Unknonws', ...
                'EdgeColor', edgeColor, ...
                'FaceAlpha', faceAlpha);
            legend();
            
            
            title('');
            xlabel('Frame')
            ylabel('Number');
            
            nf = numel(cp.NO); % number of frames
            xticks([1, this.delta, nf]);

            axis('tight');
            box('off');
            this.setFontSize();
            
            this.saveFigure('classification-unknonw-numbers');
        end
        
        function plotConfusionMatrix(this, ul)
            
            cp = this.getClassificationPerformance(ul);
            d = this.delta;
            
            y = cell2mat(cp.Y((d + 1):end));
            y_ = cell2mat(cp.Y_((d + 1):end));
            
            % DataReader.createFigure('Classification - Confusion Matrix');
            figure('Name', 'Classification - Confusion Matrix', 'Color', [1, 1, 1]);
            
            cm = confusionchart(y, y_);
            cm.ColumnSummary = 'column-normalized';
            cm.RowSummary = 'row-normalized';
            % cm.ClassLabels = {'Star', 'Object'};
            
            % plotconfusion(y', y_')
            
            if ul
                title('Unknonws are Objects');
            else
                title('Unknonws are Stars');
            end
            
            this.setFontSize();
            
            this.saveFigure(sprintf('classification-confusion-matrix-2-ul-%d', ul));
        end
    end
    
    % Viz
    methods
        function makeClassifiedVideo(this, tags)
            
            fprintf('\nMake classified video: ...\n');
            tic();
            
            vr = VideoReader(this.filenames.video);
            width = vr.Width;
            height = vr.Height;
            fps = vr.FrameRate;
            
            vw = VideoWriter(this.filenames.classifiedVideo, 'MPEG-4');
            vw.FrameRate = fps;
            open(vw);
            
            load(this.filenames.output, 'centroids', 'labels');
            
            T = numel(labels);
            
            if nargin < 2
                tags = cell(T, 1);
            end
            
            for t = 1:T
                I = zeros(height, width, 3);
                l = labels{t};
                c = centroids{t};
                
                txt = sprintf('Frame: %d, Spots: %d', t, size(c, 1));
                
                if isempty(l)
                    % spots
                    I = drawSpots(I, c, [1, 1, 1]);
                    
                    % title
                    txt = sprintf('%s, Skipped', txt);
                else
                    % unknow
                    I = drawSpots(I, c(isnan(l), :), [1, 0, 1]);
                    % stars
                    I = drawSpots(I, c(l == 0, :), [1, 1, 1]);
                    % objects
                    I = drawSpots(I, c(l == 1, :), [1, 0, 0]);

                    % title
                    txt = sprintf(...
                        '%s, Objects: %d, Unknown: %d', ...
                        txt, sum(l == 1), sum(isnan(l)));
                end
                
                % tracking
                if ~isempty(tags{t}) && ~isempty(l)
                    idx = ~isnan(tags{t}) & (l == 1);
                    
                    if any(idx)
                        position = c(idx, :);
                        text = cellstr(string(tags{t}(idx)));
                        
                        I = insertText(...
                            I, ...
                            position, ...
                            text, ...
                            'Font', 'LucidaTypewriterBold', ...
                            'TextColor', [1, 1, 1], ...
                            'BoxOpacity', 0.4);
                    end
                end
                
                I = insertText(...
                    I, ...
                    [10, 10], ...
                    txt, ...
                    'FontSize', 24, ...
                    'BoxColor', [1, 1, 0], ...
                    'BoxOpacity', 0.4, ...
                    'TextColor', [1, 1, 1]);
            
                writeVideo(vw, I);
                
                disp(txt);
            end
            
            close(vw);
            
            fprintf('\nFile `%s` saved: ', this.filenames.classifiedVideo);
            toc();
            
            implay(this.filenames.classifiedVideo);
            
            % Local functions
            function I = drawSpots(I, position, color)
                r = 5;
                opacity = 0.7;
                
                position = [position, r * ones(size(position, 1), 1)];
                
                I = insertShape(I, ...
                    'FilledCircle', position, ...
                    'Color', color, ...
                    'Opacity', opacity ...
                );
            end
        end
        
        function setFontSize(~)
            set(gca, 'FontSize', 20);
        end
        
        function saveFigure(this, name)
            savefig(fullfile(this.folders.results, name));
        end
    end
    
    % Main
    methods (Static)
        function main()
            close('all');
            clc();
            
            dataFolder = fullfile(Tracker.ASSETS_FOLDER, 'data');
            
            fprintf('\nFind optimal parameters:\n');
            fprintf('\nMake classified videos:\n');
            mainTimer = tic();
            
            listing = dir(fullfile(dataFolder, '*.mp4'));
            n = numel(listing);
            parfor i = 1:n
                filename = fullfile(listing(i).folder, listing(i).name);
                
                Tracker.findOptimalParams(filename);
                Tracker.classification(filename);
            end
            toc(mainTimer);
        end
        
        function findOptimalParams(filename)
            % Properties
            % filename = './assets/data/SampleVideo4.mp4';
            binarizationThreshold = 0.1;
            % unmatchedCost = 100;
            maxNumObjects = 10; % todo: must be `10`
            % delta = 10;
            isVelocityMagnitude = true;
            isVelocityDirection = true;
            
            
            fprintf('\nMeasure classification performance:\n');
            mainTimer = tic();
            
            ind = 1;
            for delta = Tracker.DELTA
                for unmatchedCost = Tracker.UNMATCHED_COST
                    params(ind) = struct(...
                        'delta', delta, ...
                        'unmatchedCost', unmatchedCost);
                    ind = ind + 1;
                end
            end
            
            perf = struct();
            parfor i = 1:numel(params) % todo: must be `parfor`
                delta = params(i).delta;
                unmatchedCost = params(i).unmatchedCost;

                fprintf('d: %d, \tc: %d \t', delta, unmatchedCost);
                localTimer = tic();

                try
                    tracker = Tracker(...
                        filename, ...
                        binarizationThreshold, ...
                        unmatchedCost, ...
                        maxNumObjects, ...
                        delta, ...
                        isVelocityMagnitude, ...
                        isVelocityDirection);

                    % Classification
                    tracker.classify();

                    % Classification performance
                    cp = tracker.getClassificationPerformance();
                catch
                    cp = struct(...
                        'TPR', struct('mean', nan, 'std', nan), ...
                        'TNR', struct('mean', nan, 'std', nan), ...
                        'UNK', struct('mean', nan, 'std', nan));
                end

                perf(i).d = delta;
                perf(i).c = unmatchedCost;
                perf(i).tpr = cp.TPR;
                perf(i).tnr = cp.TNR;
                perf(i).unk = cp.UNK;

                toc(localTimer);
            end
            
            [~, name] = fileparts(filename);
            resultsFolder = Tracker.getResultsFolder(filename);
            perfFilename = fullfile(resultsFolder, sprintf('%s-perf.mat', name));
            save(perfFilename, 'perf');
            Tracker.plotClassificationPerformanceOverall(perfFilename)
            
            toc(mainTimer);
        end
        
        function classification(filename)
            % Properties
            binarizationThreshold = 0.1;
            unmatchedCost = 100;
            maxNumObjects = 10;
            delta = 10;
            isVelocityMagnitude = true;
            isVelocityDirection = true;
            
            % Tracker
            tracker = Tracker(...
                filename, ...
                binarizationThreshold, ...
                unmatchedCost, ...
                maxNumObjects, ...
                delta, ...
                isVelocityMagnitude, ...
                isVelocityDirection);
            
            % Classification
            tracker.classify();
            tracker.makeClassifiedVideo();
        end
        
        function plotClassificationPerformance(perfFilename, name1, name2)
            
            load(perfFilename, 'perf');
            
            
            [D, C] = meshgrid(Tracker.DELTA, Tracker.UNMATCHED_COST);
            
            V = zeros(size(D));
            
            for i = 1:numel(perf)
                id = find(Tracker.DELTA == perf(i).d, 1);
                ic = find(Tracker.UNMATCHED_COST == perf(i).c, 1);
                
                V(ic, id) = perf(i).(name1).(name2);
            end
            
            surfc(D, C, V);
            % view([-67, 30]);
            
            % title(sprintf('%s(%s)', name2, name1));
            
            % xlabel('delay [d] (frame)');
            xlim([1, 33]);
            
            % ylabel('unmatched cost (c) (pixel)');
            set(gca, 'YScale', 'log');
            ylim([1, 64]);
            yticks(Tracker.UNMATCHED_COST);
            % yticklabels(cellstr(string(2 .^ (0:10))));
            
            zticks([]);
            
            shading('interp');
            % colorbar('eastoutside');
            
            % colormap('gray');
            
            switch name2
                case 'mean'
                    caxis([0, 1]);
                case 'std'
                    caxis([0, 0.1]);
            end
            
            % axis('tight');
            
            set(gca, 'FontSize', 36);
        end
        
        function plotClassificationPerformanceOverall(perfFilename)
            DataReader.createFigure('Classification Performance');
            
            rows = 3;
            cols = 2;
            
            % tpr
            % - mean
            subplot(rows, cols, 1);
            Tracker.plotClassificationPerformance(perfFilename, 'tpr', 'mean');
            zlabel(sprintf('Object\n'));
            c = colorbar('eastoutside');
            c.Label.String = 'TPR \{mean\}';
            c.Ticks = [0, 0.5, 1];
            % - std
            subplot(rows, cols, 2);
            Tracker.plotClassificationPerformance(perfFilename, 'tpr', 'std');
            c = colorbar('eastoutside');
            c.Label.String = 'TPR \{std\}';
            c.Ticks = [0, 0.05, 0.1];
            % tnr
            % - mean
            subplot(rows, cols, 3);
            Tracker.plotClassificationPerformance(perfFilename, 'tnr', 'mean');
            zlabel(sprintf('Star\n'));
            c = colorbar('eastoutside');
            c.Label.String = 'TNR \{mean\}';
            c.Ticks = [0, 0.5, 1];
            % - std
            subplot(rows, cols, 4);
            Tracker.plotClassificationPerformance(perfFilename, 'tnr', 'std');
            c = colorbar('eastoutside');
            c.Label.String = 'TNR \{std\}';
            c.Ticks = [0, 0.05, 0.1];
            % unk
            % - mean
            subplot(rows, cols, 5);
            Tracker.plotClassificationPerformance(perfFilename, 'unk', 'mean');
            xlabel('delay [d] (frame)');
            ylabel('unmatched cost [c] (pixel)');
            zlabel(sprintf('Unknonw\n'));
            c = colorbar('eastoutside');
            c.Label.String = 'Unknonw rate \{mean\}';
            c.Ticks = [0, 0.5, 1];
            % - std
            subplot(rows, cols, 6);
            Tracker.plotClassificationPerformance(perfFilename, 'unk', 'std');
            c = colorbar('eastoutside');
            c.Label.String = 'Unknonw rate \{std\}';
            caxis([0, 0.02]);
            c.Ticks = [0, 0.01, 0.02];

            [folder, name] = fileparts(perfFilename);
            savefig(fullfile(folder, [name, '.fig']));
            saveas(gcf, fullfile(folder, [name, '.png']));
        end
    end
    
    % Helper
    methods (Static)
        function nan2one()
            filename = './assets/results/output.mat';
            
            load(filename, 'labels');
            
            for i = 1:numel(labels)
                labels{i}(isnan(labels{i})) = 1;
            end
            
            save(filename, 'labels', '-append');
        end
        
        function resultsFolder = getResultsFolder(filename)
            
            [~, name] = fileparts(filename);
            resultsFolder = fullfile(Tracker.ASSETS_FOLDER, 'results', name);    
            
            if ~exist(resultsFolder, 'dir')
                mkdir(resultsFolder);
            end
        end
    end
end

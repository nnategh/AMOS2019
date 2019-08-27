classdef DataReader < handle
    % Data Reader
    
    % Properties
    properties
        % - folders: struct
        %   Path of folders
        % - filenames: struct
        %   Path of filenames
        
        % - N: number
        %   Number of frames
        
        % - data: struct
        %   Sample data
        
        % - idx: struct(...
        %   'spot': number[frame][spot]
        %   'obj': number[frame][object]
        %   'star': number[frame][star]
        %   )
        %   Indeces of spots, objects, and stars for each frame
        
        % - flux: struct(...
        %   'spot': number[frame][spot]
        %   'obj': number[frame][object]
        %   'star': number[frame][star]
        %   )
        %   Photon flux of spots, objects, and stars for each frame
        
        % - mag: struct(...
        %   'spot': number[frame][spot]
        %   'obj': number[frame][object]
        %   'star': number[frame][star]
        %   )
        %   Star magnitude flux of spots, objects, and stars for each frame
        
        % - pp: struct(...
        %   'spot': number[frame][spot, 2]
        %   'obj': number[frame][object, 2]
        %   'star': number[frame][star, 2]
        %   )
        %   Shuffled pixel positions of spots, objects, and stars for each frame
        
        % - lbl: struct(...
        %   'spot': number[frame][spot, 2]
        %   'obj': number[frame][object, 2]
        %   'star': number[frame][star, 2]
        %   )
        %   Labels of spots, objects, and stars for each frame
        
        folders
        filenames
        
        data
        
        N
        
        idx
        
        flux
        
        mag
        
        pp
        
        lbl
    end
    
    % Constructor
    methods
        function this = DataReader(filename)
            % todo: there is no need to `videoFilename`
            
            % folders/filenames
            this.initFolders();
            this.initFilenames(filename);
            
            % data
            this.initData();
            
            % number of frames
            this.initN();
            
            % indeces
            this.initIdx();
            
            % photon flux
            this.initFlux();
            
            % star magnitude
            this.initMag();
            
            % pixel position
            this.initPP();
            
            % object labels
            this.initLbl();
        end
        
        % Folders/Filenames
        function initFolders(this)
            assetsFolder = './assets';
            
            this.folders = struct(...
                'assets', assetsFolder, ...
                'results', fullfile(assetsFolder, 'results'));
            
            % assets
            if ~exist(this.folders.assets, 'dir')
                mkdir(this.folders.assets);
            end
            
            % results
            if ~exist(this.folders.results, 'dir')
                mkdir(this.folders.results);
            end
        end
        
        function initFilenames(this, filename)
            [~, name] = fileparts(filename);
            
            this.filenames = struct(...
                'data', filename, ...
                'video', fullfile(this.folders.results, [name, '-video.mp4']), ...
                'isObjFigure', fullfile(this.folders.results, [name, '-isobj.fig']), ...
                'fluxHistFigure', fullfile(this.folders.results, [name, '-flux-hist.fig']), ...
                'fluxBoxFigure', fullfile(this.folders.results, [name, '-flux-box.fig']), ...
                'magHistFigure', fullfile(this.folders.results, [name, '-mag-hist.fig']), ...
                'magBoxFigure', fullfile(this.folders.results, [name, '-mag-box.fig']));
        end
        
        % Data
        function initData(this)
            this.data = getfield(...
                load(this.filenames.data), ...
                'SampleData');
        end
        
        % Number of frames
        function initN(this)
            times = this.data.AlgorithmInputData.Time; % [spot, frame]
            this.N = numel(times);
        end
        
        % Indeces
        function initIdx(this)
            pf = this.data.AlgorithmInputData.Spot_PhotonFlux; % [spot, frame]
            io = logical(this.data.AlgorithmInputData.Is_Object_TrainingData); % [spot, frame]
            
            n = this.N;
            this.idx.spot = cell(n, 1);
            this.idx.obj = cell(n, 1);
            this.idx.star = cell(n, 1);
            for i = 1:n
                this.idx.spot{i} = find(pf(:, i));
                this.idx.obj{i} = find(io(this.idx.spot{i}, i) > 0);
                this.idx.star{i} = find(io(this.idx.spot{i}, i) == 0);
            end
        end
        
        % Photon Flux
        function initFlux(this)
            pf = this.data.AlgorithmInputData.Spot_PhotonFlux;
            
            n = this.N;
            this.flux.spot = cell(n, 1);
            this.flux.obj = cell(n, 1);
            this.flux.star = cell(n, 1);
            
            for i = 1:n
                this.flux.spot{i} = pf(this.idx.spot{i}, i);
                this.flux.obj{i} = pf(this.idx.obj{i}, i);
                this.flux.star{i} = pf(this.idx.star{i}, i);
            end
        end
        
        % Star Magnitude
        function initMag(this)
            sm = this.data.AlgorithmInputData.Spot_StarMagnitude;
            
            n = this.N;
            this.mag.spot = cell(n, 1);
            this.mag.obj = cell(n, 1);
            this.mag.star = cell(n, 1);
            
            for i = 1:n
                this.mag.spot{i} = sm(this.idx.spot{i}, i);
                this.mag.obj{i} = sm(this.idx.obj{i}, i);
                this.mag.star{i} = sm(this.idx.star{i}, i);
            end
        end
        
        % Pixel Position
        function initPP(this)
            xy = this.data.AlgorithmInputData.Spot_CentroidPositions_XY; % [position, spot, frame]
            xy = permute(xy, [2, 1, 3]); % [spot, position, frame]
            
            n = this.N;
            this.pp.spot = cell(n, 1);
            this.pp.obj = cell(n, 1);
            this.pp.star = cell(n, 1);
            
            for i = 1:n
                this.pp.spot{i} = xy(this.idx.spot{i}, :, i);
                this.pp.obj{i} = xy(this.idx.obj{i}, :, i);
                this.pp.star{i} = xy(this.idx.star{i}, :, i);
            end
        end
        
        % Object labels
        function initLbl(this)
            
            io = this.data.AlgorithmInputData.Is_Object_TrainingData; % [spots, time]
            
            n = this.N;
            this.lbl.obj = cell(n, 1);
            
            for i = 1:n
                this.lbl.obj{i} = io(this.idx.obj{i}, i);
            end
        end
    end
    
    % Is object
    methods
        function plotIsObj(this)
            % show isObj
            
            % todo: 
            % - some pixels must be nan (number of spots are not the
            %   same during frams)
            %   star: blue, object: red, nothing: white
            
            rows = 2;
            cols = 1;
            
            DataReader.createFigure('Is object');
            
            % image
            subplot(rows, cols, 1);
            
            io = logical(this.data.AlgorithmInputData.Is_Object_TrainingData);
            imagesc(io);
            
            colormap(gca, [[0, 0, 0]; [1, 1, 1]]);

            % ticks
            % - x
            % ns: number of spots
            % nf: number of times
            [ns, nf] = size(io);
            xticks([1, nf]);
            % - y
            yticks([1, ns]);

            title('SampleData.AlgorithmInputData.Is\_Object\_TrainingData');
            xlabel('Frame')
            ylabel('Spot index');

            axis('tight');
            axis('equal');

            this.setAxis();
            set(gca(), 'GridColor', [1, 1, 1]);
            
            % plot
            no = sum(io); % number of objects
            
            subplot(rows, cols, 2);
            area(no);
            
            xticks([1, nf]);
            yticks([0, max(no)]);

            xlabel('Frame')
            ylabel('Number of objects');

            axis('tight');
            this.setAxis();
            box('off');
            
            savefig(this.filenames.isObjFigure);
        end
    end
    
    % Photon Flux
    methods
        function histFlux(this)
            [FO, FS] = this.getFlux();

            % mean +- std
            fprintf('Object: %g (%g)\n', mean(FO), std(FO));
            fprintf('Star: %g (%g)\n', mean(FS), std(FS));

            DataReader.createFigure('Phton Flux');
            histogram(FO, 'Normalization', 'probability', 'FaceAlpha', 0.5);
            hold('on');
            histogram(FS, 'Normalization', 'probability', 'FaceAlpha', 0.5);

            title('SampleData.AlgorithmInputData.Spot\_PhotonFlux');
            xlabel('log_{10}(Photon flux)');
            xticks(round(sort([
                min(FO)
                mean(FO)
                max(FO)
                min(FS)
                mean(FS)
                max(FS)]), 1));
            
            yticks(0:0.1:1);
            ylim([0, 0.2]);
            ylabel('Probability');
            legend('Object', 'Star');
            
            this.setAxis();
            
            savefig(this.filenames.fluxHistFigure);
        end
        
        function boxFlux(this)
            [FO, FS] = this.getFlux();

            % mean +- std
            fprintf('Object: [%g, %g, %g]\n', min(FO), median(FO), max(FO));
            fprintf('Star: [%g, %g, %g]\n', min(FS), median(FS), max(FS));

            DataReader.createFigure('Phton Flux');
            boxplot([FO;FS], [zeros(numel(FO), 1); ones(numel(FS), 1)],...
                'Notch', 'off', ...
                'Labels', {'Object','Star'});

            title('SampleData.AlgorithmInputData.Spot\_PhotonFlux');
            ylabel('log_{10}(Photon flux)');
            yticks(round(sort([
                min(FO)
                median(FO)
                max(FO)
                min(FS)
                median(FS)
                max(FS)]), 2));
            
            this.setAxis();
            
            savefig(this.filenames.fluxBoxFigure);
        end
        
        function [FO, FS] = getFlux(this)
            % change to column vector
            FO = cell2mat(this.flux.obj);
            FS = cell2mat(this.flux.star);

            % log base 10
            FO = log10(FO);
            FS = log10(FS);
        end
    end
    
    % Photon Flux
    methods
        function histMag(this)
            [MO, MS] = this.getMag();

            % mean +- std
            fprintf('Object: %g (%g)\n', mean(MO), std(MO));
            fprintf('Star: %g (%g)\n', mean(MS), std(MS));

            DataReader.createFigure('Star Magnitude');
            histogram(MO, 'Normalization', 'probability', 'FaceAlpha', 0.5);
            hold('on');
            histogram(MS, 'Normalization', 'probability', 'FaceAlpha', 0.5);

            title('SampleData.AlgorithmInputData.Spot\_StarMagnitude');
            xlabel('Star magnitude');
            xticks(round(sort([
                min(MO)
                mean(MO)
                max(MO)
                min(MS)
                mean(MS)
                max(MS)]), 1));
            
            yticks(0:0.1:1);
            ylim([0, 0.2]);
            ylabel('Probability');
            legend('Object', 'Star');
            
            this.setAxis();
            
            savefig(this.filenames.magHistFigure);
        end
        
        function boxMag(this)
            [MO, MS] = this.getMag();

            % mean +- std
            fprintf('Object: [%g, %g, %g]\n', min(MO), median(MO), max(MO));
            fprintf('Star: [%g, %g, %g]\n', min(MS), median(MS), max(MS));

            DataReader.createFigure('Star Magnitude');
            boxplot([MO;MS], [zeros(numel(MO), 1); ones(numel(MS), 1)],...
                'Notch', 'off', ...
                'Labels', {'Object','Star'});

            title('SampleData.AlgorithmInputData.Spot\_StarMagnitude');
            ylabel('Star magnitude');
            yticks(round(sort([
                min(MO)
                median(MO)
                max(MO)
                min(MS)
                median(MS)
                max(MS)]), 2));
            
            this.setAxis();
            
            savefig(this.filenames.magBoxFigure);
        end
        
        function [MO, MS] = getMag(this)
            % change to column vector
            MO = cell2mat(this.mag.obj);
            MS = cell2mat(this.mag.star);
        end
    end
    
    % Pixel Position
    methods
        function makeVideo(this)
            % Dilate spots in sample video
            
            r = 5; % spot radius
            
            fprintf('\nMake `%s` video:\n', this.filenames.video);
            tic();
            
            [width, height, fps] = this.getVideoInfo();
            pos = this.pp.spot; % spot pixel positions
            
            fprintf('Width: %d\n', width);
            fprintf('Height: %d\n', height);
            fprintf('Frames: %d\n', numel(pos));
            fprintf('Frame rate: %g\n', fps);
            
            vw = VideoWriter(this.filenames.video, 'MPEG-4');
            vw.FrameRate = fps;
            open(vw);
            
            for t = 1:numel(pos)
                position = pos{t};
                position = [position(:, [1, 2]), r * ones(size(position, 1), 1)];
                
                I = zeros(height, width, 3);
                I = insertShape(I, ...
                    'FilledCircle', position, ...
                    'Color', [1, 1, 1], ...
                    'Opacity', 1 ...
                );
            
                writeVideo(vw, I);
                
                fprintf('Frame: %d, \tSpots: %d\n', t, size(position, 1));
            end
            
            close(vw);
            toc();
            
            implay(this.filenames.video);
        end
        
        function [width, height, fps] = getVideoInfo(this)
            width = this.data.NumberPixels(1);
            height = this.data.NumberPixels(2);
            fps = round(1 / this.data.DT);
        end
        
        function trueVsInputObjPixel(this)
            rows = 1;
            cols = 2;
            r = 10;
            color = [255, 0, 0];
            opacity = 0.4;
            
            % true position
            P = this.trueObjPixel;
            
            % max time of appearance
            fprintf('Max time of appearance:\n');
            fprintf('  True: %d, %d\n', ...
                find(isnan(P(1, :, 1)), 1), ...
                find(isnan(P(2, :, 1)), 1));
            
            % input pixel positions
            P_ = this.objPixel;
            
            fprintf('  Input: %d, %d\n', ...
                find(isnan(P_(1, :, 1)), 1), ...
                find(isnan(P_(2, :, 1)), 1));
            
            vr = VideoReader(this.filenames.video);
            vw = VideoWriter(this.filenames.trueVsInputObjPixel, 'MPEG-4');
            vw.FrameRate = vr.FrameRate;
            open(vw);
            
            t = 1;
            fig = DataReader.createFigure('True vs input pixel position');
            while hasFrame(vr)
                frame = readFrame(vr);
                
                % true
                I = frame;
                
                position = squeeze(P(:, t, :));
                position = position(any(~isnan(position), 2), :);
                if ~isempty(position)
                    position = [position, r * ones(size(position, 1), 1)];

                    I = insertShape(I, ...
                        'FilledCircle', position, ...
                        'Color', color, ...
                        'Opacity', opacity ...
                    );
                end
                
                subplot(rows, cols, 1);
                imshow(I);
                title('SampleData.Objects.Centroid\_XY');
                
                % input
                I = frame;
                
                position = squeeze(P_(:, t, :));
                position = position(any(~isnan(position), 2), :);
                if ~isempty(position)
                    position = [position, r * ones(size(position, 1), 1)];

                    I = insertShape(I, ...
                        'FilledCircle', position, ...
                        'Color', color, ...
                        'Opacity', opacity ...
                    );
                end
                
                subplot(rows, cols, 2);
                imshow(I);
                title('SampleData.AlgorithmInputData.Spot\_CentroidPositions\_XY');
                
                suptitle(sprintf('Frame #%3d', t));
                
                writeVideo(vw, getframe(fig));
                
                t = t + 1;
            end
            
            close(vw);
        end
        
        function spotPixelOverDilatedVideo(this, trueFlag, labelFlag)
            if nargin < 2
                trueFlag = false;
            end
            
            if nargin < 3
                labelFlag = false;
            end
            
            r = 10;
            starColor = [0, 0, 255];
            objColor = [255, 0, 0];
            opacity = 0.4;
            
            % input pixel positions
            if trueFlag
                SP = this.trueStarPixel;
                OP = this.trueObjPixel;
            else
                SP = this.pp.star;
                OP = this.pp.obj;
            end
            
            % input video
            filename = this.filenames.video;
            fprintf('\nLoad `%s` video file.\n', filename);
            vr = VideoReader(this.filenames.video);
            fprintf('Width: %d\n', vr.Width);
            fprintf('Height: %d\n', vr.Height);
            fprintf('Duration: %g\n', vr.Duration);
            fprintf('Frame rate: %g\n', vr.FrameRate);
            
            % output video
            [~, name, ext] = fileparts(filename);
            if trueFlag
                name = [name, '-true'];
            else
                name = [name, '-shuffled'];
            end
            if labelFlag
                name = [name, '-labeled'];
            end
            filename = fullfile(this.folders.results, [name, ext]);
            
            vw = VideoWriter(filename, 'MPEG-4');
            vw.FrameRate = vr.FrameRate;
            open(vw);
            
            fprintf('\nMake `%s` video:\n', filename);
            tic();
            
            t = 1;
            
            % figure title
            figTitle = 'spot pixel position over dilated video';
            if trueFlag
                figTitle = ['true ', figTitle];
            else
                figTitle = ['shuffled ', figTitle];
            end
            if labelFlag
                figTitle = ['labelled ', figTitle];
            end
            figTitle = [upper(figTitle(1)), figTitle(2:end)];
            
%             DataReader.createFigure(figTitle);
            
            while hasFrame(vr)
                frame = readFrame(vr);
                
                % star
                % S = squeeze(SP(:, t, :));
                % frame = drawOnFrame(frame, S, starColor);
                
                % opject
                % O = squeeze(OP(:, t, :));
                labels = cellstr(string(this.lbl.obj{t}));
                frame = drawOnFrame(frame, OP{t}, objColor, labels);
                
                % title
                % NS = sum(~isnan(S(:, 1)));
                % NO = sum(~isnan(O(:, 1)));
                
                NS = size(SP{t}, 1);
                NO = size(OP{t}, 1);
                
                txt = sprintf(...
                        'Frame: %3d, Spots: %3d, Objects: %d', ...
                        t, NS + NO, NO);
                disp(txt);
                frame = insertText(...
                    frame, ...
                    [10, 10], ...
                    txt, ...
                    'FontSize', 24, ...
                    'BoxColor', [255, 255, 0], ...
                    'BoxOpacity', 0.4, ...
                    'TextColor', [255, 255, 255]);
                
%                 imshow(frame);
                writeVideo(vw, frame);
                
                t = t + 1;
            end
            
            close(vw);
            toc();
            
            implay(filename);
            
            % Local functions
            function frame = drawOnFrame(frame, position, color, labels)
                position = position(any(~isnan(position), 2), :);
                n = size(position, 1);
                
                if ~isempty(position)
                    if labelFlag
                        frame = insertText(...
                            frame, ...
                            position, ...
                            labels, ...
                            'Font', 'LucidaTypewriterBold', ...
                            'TextColor', color, ...
                            'BoxOpacity', opacity);
                    else
                        position = [position, repmat(r, [n, 1])];
                        frame = insertShape(...
                            frame, ...
                            'FilledCircle', position, ...
                            'Color', color, ...
                            'Opacity', opacity);
                    end
                end
            end
        end
    end
    
    % Unit Vector(Attitude) Position
    methods
        function histTrueUnitVecDiffAng(this, earthFlag)
            if earthFlag
                AS = this.trueStarUnitVecDiffAngEarth;
                AO = this.trueObjUnitVecDiffAngEarth;
            else
                AS = this.trueStarUnitVecDiffAngCam;
                AO = this.trueObjUnitVecDiffAngCam;
            end
            

            % mean +- std
            fprintf('Star: %g (%g)\n', mean(AS), std(AS));
            fprintf('Object: %g (%g)\n', mean(AO), std(AO));
            

            DataReader.createFigure('Unit Vector(Attitude) - Difference of Angles');
            histogram(AS, 'Normalization', 'probability', 'FaceAlpha', 0.5);
            hold('on');
            histogram(AO, 'Normalization', 'probability', 'FaceAlpha', 0.5);

            if earthFlag
                title('SampleData.Stars.UnitVec\_EarthFrame | SampleData.Objects.UnitVec\_EarthFrame');
            else
                title('SampleData.Stars.UnitVec\_CamFrame | SampleData.Objects.UnitVec\_CamFrame');
            end
            xlabel('Difference angle of unit vectors in the camera frame');
            xticks(unique(round(sort([
                min(AS)
                mean(AS)
                max(AS)
                min(AO)
                mean(AO)
                max(AO)]), 4)));
            
            ylabel('Probability');
            yticks(0:0.1:1);
            ylim([0, 1]);
            legend('Star', 'Object');
            
            this.setAxis();
            
            if earthFlag
                savefig('assets/results/true_unit_vect_diff_ang_cam');
            else
                savefig('assets/results/true_unit_vect_diff_ang_earth');
            end
        end
    end
    
    % Helper methods
    methods
        function setAxis(~)
            box('on');
            grid('on');
            set(gca(), 'FontSize', 20);
        end
    end
    methods(Static)
        function a = ang(u1, u2)
            % for values outside the interval [-1,1] `acos` returns complex values
            a = abs(acos(dot(u1, u2)));
        end
        
        function h = createFigure(name)
            % Create `full screen` figure
            %
            % Parameters
            % ----------
            % - name: string
            %   Name of figure
            %
            % Return
            % - h: matlab.ui.Figure
            %   Handle of created figure

            h = figure(...
                'Name', name, ...
                'Color', 'white', ...
                'NumberTitle', 'off', ...
                'Units', 'normalized', ...
                'OuterPosition', [0, 0, 1, 1] ...
            );
        end
    end
    
    % Main
    methods(Static)
        function main()
            close('all');
            clc();
            
            filename = './assets/data/complex-fg.mat';
            dr = DataReader(filename);
            
            % is object
            dr.plotIsObj();
            
            % photon flux
            fprintf('--- Photon flux ---\n');
            dr.histFlux();
            dr.boxFlux();
             
            % star magnitude
            fprintf('\n--- Star magnitude ---\n');
            dr.histMag();
            dr.boxMag();
            
            % pixel position
            dr.makeVideo();
            dr.spotPixelOverDilatedVideo(false, false);
            dr.spotPixelOverDilatedVideo(false, true);
            dr.spotPixelOverDilatedVideo(true, false);
            dr.spotPixelOverDilatedVideo(true, true);
        end
        
        function amos19()
            close('all');
            clc();

            % filename = './assets/data/simple-fg.mat';
            filename = './assets/data/complex-fg.mat';
            % filename = './assets/data/simple-bg.mat';
            % filename = './assets/data/complex-bg.mat';
            dr = DataReader(filename);

            % Spots
            spot = dr.idx.spot;
            nf = numel(spot); % number of frames
            ns = zeros(nf, 1); % number of spots in each frame
            for i = 1:nf
                ns(i) = numel(spot{i});
            end
            fprintf('Number of spots per frame: %.1f $\\pm$ %.1f\n', mean(ns), std(ns));
            
            % Objects
            obj = dr.idx.obj;
            nf = numel(obj); % number of frames
            no = zeros(nf, 1); % number of spots in each frame
            for i = 1:nf
                no(i) = numel(obj{i});
            end
            fprintf('Number of objects per frame: %.1f $\\pm$ %.1f\n', mean(no), std(no));
        end
    end
end


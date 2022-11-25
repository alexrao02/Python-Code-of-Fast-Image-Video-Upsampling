function [H_tilde_Y,H_Cb,H_Cr,H_bicubic] = getbicubic()
IMG.name = 'chip';
IMG.read = [IMG.name, '_input.png'];
L_RGB = im2double(imread(IMG.read));
L_YCbCr = rgb2ycbcr(L_RGB);
UPSAMP.factor = 4;
H_tilde_Y = im2double(imresize(L_YCbCr(:,:,1), UPSAMP.factor, 'bicubic'));
H_Cb = im2double(imresize(L_YCbCr(:,:,2), UPSAMP.factor, 'bicubic'));
H_Cr = im2double(imresize(L_YCbCr(:,:,3), UPSAMP.factor, 'bicubic'));
H_bicubic = cat(3, H_tilde_Y, H_Cb, H_Cr);
H_bicubic = ycbcr2rgb(H_bicubic);
save('H_tilde_Y.mat','H_tilde_Y')
save('H_Cb.mat','H_Cb')
save('H_Cr.mat','H_Cr')
save('H_bicubic.mat','H_bicubic')
end


def backwarp(img, flow):
  _, _, H, W = img.size()
  u = flow[:, 0, :, :]
  v = flow[:, 1, :, :]
  gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
  gridX = torch.tensor(gridX, requires_grad=False,).cuda()
  gridY = torch.tensor(gridY, requires_grad=False,).cuda()
  x = gridX.unsqueeze(0).expand_as(u).float() + u
  y = gridY.unsqueeze(0).expand_as(v).float() + v
  # range -1 to 1
  x = 2*(x/W - 0.5)
  y = 2*(y/H - 0.5)
  # stacking X and Y
  grid = torch.stack((x,y), dim=3)
  # Sample pixels using bilinear interpolation.
  imgOut = torch.nn.functional.grid_sample(img, grid)
  return imgOut

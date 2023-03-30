import os.path as osp
import logging


class BaseImageDataset(object):

    def print_dataset_statistics(self, args, train, query, gallery):
        _, name = osp.split(args.output_dir)
        logger = logging.getLogger("{}.dataset".format(name))
               
        num_vids, num_imgs, num_cams = self.get_imagedata_info(train)
        num_vids_q, num_imgs_q, num_cams_q = self.get_imagedata_info(query)
        num_vids_g, num_imgs_g, num_cams_g = self.get_imagedata_info(gallery)

        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_vids, \
                                                                num_imgs, \
                                                                num_cams))
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_vids_q, \
                                                                num_imgs_q, \
                                                                num_cams_q))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_vids_g, \
                                                                num_imgs_g, \
                                                                num_cams_g))
        logger.info("  ----------------------------------------")
    
    def get_imagedata_info(self, data):
        vids, cams = [], []
        for _, vid, camid in data:
            vids += [vid]
            cams += [camid]
            
        vids, cams = set(vids), set(cams)
        return len(vids), len(data), len(cams)
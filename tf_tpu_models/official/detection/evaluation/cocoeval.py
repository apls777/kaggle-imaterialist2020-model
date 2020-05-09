from pycocotools import cocoeval
import numpy as np


class COCOeval(cocoeval.COCOeval):

    def summarize_per_category(self):
        '''
        Compute and display summary metrics for evaluation results *per category*.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize_single_category(ap=1, iouThr=None, categoryId=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ CategoryId={:>3d} | IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if categoryId is not None:
                    category_index = [i for i, i_catId in enumerate(p.catIds) if i_catId == categoryId]
                    s = s[:,:,category_index,aind,mind]
                else:
                    s = s[:,:,:, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if categoryId is not None:
                    category_index = [i for i, i_catId in enumerate(p.catIds) if i_catId == categoryId]
                    s = s[:,category_index,aind,mind]
                else:
                    s = s[:,:, aind, mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            #print(iStr.format(titleStr, typeStr, catId, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets_per_category():
            category_stats = np.zeros((12,len(self.params.catIds)))
            for category_index, category_id in enumerate(self.params.catIds):
                category_stats[0][category_index] = _summarize_single_category(1,
                                                                               categoryId=category_id)
                category_stats[1][category_index] = _summarize_single_category(1,
                                                                               iouThr=.5,
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[2][category_index] = _summarize_single_category(1,
                                                                               iouThr=.75,
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[3][category_index] = _summarize_single_category(1,
                                                                               areaRng='small',
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[4][category_index] = _summarize_single_category(1,
                                                                               areaRng='medium',
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[5][category_index] = _summarize_single_category(1,
                                                                               areaRng='large',
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[6][category_index] = _summarize_single_category(0,
                                                                               maxDets=self.params.maxDets[0],
                                                                               categoryId=category_id)
                category_stats[7][category_index] = _summarize_single_category(0,
                                                                               maxDets=self.params.maxDets[1],
                                                                               categoryId=category_id)
                category_stats[8][category_index] = _summarize_single_category(0,
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[9][category_index] = _summarize_single_category(0,
                                                                               areaRng='small',
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[10][category_index] = _summarize_single_category(0,
                                                                                areaRng='medium',
                                                                                maxDets=self.params.maxDets[2],
                                                                                categoryId=category_id)
                category_stats[11][category_index] = _summarize_single_category(0,
                                                                                areaRng='large',
                                                                                maxDets=self.params.maxDets[2],
                                                                                categoryId=category_id)
            return category_stats

        def _summarizeKps_per_category():
            category_stats = np.zeros((10,len(self.params.catIds)))
            for category_index, category_id in self.params.catIds:
                category_stats[0][category_index] = _summarize_single_category(1,
                                                                               maxDets=20,
                                                                               categoryId=category_id)
                category_stats[1][category_index] = _summarize_single_category(1,
                                                                               maxDets=20,
                                                                               iouThr=.5,
                                                                               categoryId=category_id)
                category_stats[2][category_index] = _summarize_single_category(1,
                                                                               maxDets=20,
                                                                               iouThr=.75,
                                                                               categoryId=category_id)
                category_stats[3][category_index] = _summarize_single_category(1,
                                                                               maxDets=20,
                                                                               areaRng='medium',
                                                                               categoryId=category_id)
                category_stats[4][category_index] = _summarize_single_category(1,
                                                                               maxDets=20,
                                                                               areaRng='large',
                                                                               categoryId=category_id)
                category_stats[5][category_index] = _summarize_single_category(0,
                                                                               maxDets=20,
                                                                               categoryId=category_id)
                category_stats[6][category_index] = _summarize_single_category(0,
                                                                               maxDets=20,
                                                                               iouThr=.5,
                                                                               categoryId= category_id)
                category_stats[7][category_index] = _summarize_single_category(0,
                                                                               maxDets=20,
                                                                               iouThr=.75,
                                                                               categoryId=category_id)
                category_stats[8][category_index] = _summarize_single_category(0,
                                                                               maxDets=20,
                                                                               areaRng='medium',
                                                                               categoryId=category_id)
                category_stats[9][category_index] = _summarize_single_category(0,
                                                                               maxDets=20,
                                                                               areaRng='large',
                                                                               categoryId=category_id)
            return category_stats

        if not self.eval:
            raise Exception('Please run accumulate() first')

        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize_per_category = _summarizeDets_per_category
        elif iouType == 'keypoints':
            summarize_per_category = _summarizeKps_per_category
        else:
            raise ValueError('Unknown IOU type')

        self.category_stats = summarize_per_category()

    def __str__(self):
        self.summarize_per_category()

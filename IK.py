import numpy as np
import scipy.linalg as linalg
import Animation
from Quaternions import Quaternions


def joints(parents):
    return np.arange(len(parents), dtype=int)

def mask(parents, filter):
    m = np.zeros((len(parents), len(parents))).astype(bool)
    jnts = joints(parents)
    fltr = filter(parents)
    for i, f in enumerate(fltr): m[i, :] = np.any(jnts[:, np.newaxis] == f[np.newaxis, :], axis=1)
    return m

def children_list(parents):
    def joint_children(i):
        return [j for j, p in enumerate(parents) if p == i]
    return list(map(lambda j: np.array(joint_children(j)), joints(parents)))

def descendants_list(parents):
    children = children_list(parents)
    def joint_descendants(i):
        return sum([joint_descendants(j) for j in children[i]], list(children[i]))
    return list(map(lambda j: np.array(joint_descendants(j)), joints(parents)))





class IK:
    def __init__(self, animation, targets,
        references=None, iterations=10,
        recalculate=True, damping=2.0,
        secondary=0.25, translate=False,
        silent=True, weights=None,
        weights_translate=None):
        
        self.animation = animation
        self.targets = targets
        self.references = references
        
        self.iterations  = iterations
        self.recalculate = recalculate
        self.damping   = damping
        self.secondary   = secondary
        self.translate   = translate
        self.silent      = silent
        self.weights     = weights
        self.weights_translate = weights_translate
        
    def cross(self, a, b):
        o = np.empty(b.shape)
        o[...,0] = a[...,1]*b[...,2] - a[...,2]*b[...,1]
        o[...,1] = a[...,2]*b[...,0] - a[...,0]*b[...,2]
        o[...,2] = a[...,0]*b[...,1] - a[...,1]*b[...,0]
        return o
        
    def jacobian(self, x, fp, fr, ts, dsc, tdsc):
        prs = fr[:,self.animation.parents]
        prs[:,0] = Quaternions.id((1))

        tps = fp[:,np.array(list(ts.keys()))]

        qys = Quaternions.from_angle_axis(x[:,1:prs.shape[1]*3:3], np.array([[[0,1,0]]]))
        qzs = Quaternions.from_angle_axis(x[:,2:prs.shape[1]*3:3], np.array([[[0,0,1]]]))

        es = np.empty((len(x),fr.shape[1]*3, 3))
        es[:,0::3] = ((prs * qzs) * qys) * np.array([[[1,0,0]]])
        es[:,1::3] = ((prs * qzs) * np.array([[[0,1,0]]]))
        es[:,2::3] = ((prs * np.array([[[0,0,1]]])))

        j = fp.repeat(3, axis=1)
        j = dsc[np.newaxis,:,:,np.newaxis] * (tps[:,np.newaxis,:] - j[:,:,np.newaxis])
        j = self.cross(es[:,:,np.newaxis,:], j)
        j = np.swapaxes(j.reshape((len(x), fr.shape[1]*3, len(ts)*3)), 1, 2)
        
        return j

    def __call__(self, descendants=None, gamma=1.0):
        
        self.descendants = descendants

        if self.weights is None:
            self.weights = np.ones(self.animation.shape[1])
            
        if self.weights_translate is None:
            self.weights_translate = np.ones(self.animation.shape[1])

        if self.descendants is None:
            self.descendants = mask(self.animation.parents, descendants_list)
        
        self.tdescendants = np.eye(self.animation.shape[1]) + self.descendants
        
        self.first_descendants = self.descendants[:,np.array(list(self.targets.keys()))].repeat(3, axis=0).astype(int)
        self.first_tdescendants = self.tdescendants[:,np.array(list(self.targets.keys()))].repeat(3, axis=0).astype(int)

        self.endeff = np.array(list(self.targets.values()))
        self.endeff = np.swapaxes(self.endeff, 0, 1) 
        
        if not self.references is None:
            self.second_descendants = self.descendants.repeat(3, axis=0).astype(int)
            self.second_tdescendants = self.tdescendants.repeat(3, axis=0).astype(int)
            self.second_targets = dict([(i, self.references[:,i]) for i in xrange(self.references.shape[1])])
        
        nf = len(self.animation)
        nj = self.animation.shape[1]
        
        if not self.silent:
            gp = Animation.positions_global(self.animation)
            gp = gp[:,np.array(list(self.targets.keys()))]            
            error = np.mean(np.sqrt(np.sum((self.endeff - gp)**2.0, axis=2)))
            print('[IK] Start | Error: %f' % error)
        
        for i in range(self.iterations):

            gt = Animation.transforms_global(self.animation)
            gp = gt[:,:,:,3]
            gp = gp[:,:,:3] / gp[:,:,3,np.newaxis]
            gr = Quaternions.from_transforms(gt)
            
            x = self.animation.rotations.euler().reshape(nf, -1)
            w = self.weights.repeat(3)
            
            if self.translate:
                x = np.hstack([x, self.animation.positions.reshape(nf, -1)])
                w = np.hstack([w, self.weights_translate.repeat(3)])

            if self.recalculate or i == 0:
                j = self.jacobian(x, gp, gr, self.targets, self.first_descendants, self.first_tdescendants)

            l = self.damping * (1.0 / (w + 0.001))
            d = (l*l) * np.eye(x.shape[1])
            e = gamma * (self.endeff.reshape(nf,-1) - gp[:,np.array(list(self.targets.keys()))].reshape(nf, -1))
            
            x += np.array(list(map(lambda jf, ef:
                linalg.lu_solve(linalg.lu_factor(jf.T.dot(jf) + d), jf.T.dot(ef)), j, e)))

            if self.references is not None:
                
                ns = np.array(list(map(lambda jf:
                    np.eye(x.shape[1]) - linalg.solve(jf.T.dot(jf) + d, jf.T.dot(jf)), j)))
                    
                if self.recalculate or i == 0:
                    j2 = self.jacobian(x, gp, gr, self.second_targets, self.second_descendants, self.second_tdescendants)
                        
                e2 = self.secondary * (self.references.reshape(nf, -1) - gp.reshape(nf, -1))
                
                x += np.array(list(map(lambda nsf, j2f, e2f:
                    nsf.dot(linalg.lu_solve(linalg.lu_factor(j2f.T.dot(j2f) + d), j2f.T.dot(e2f))), ns, j2, e2)))

            self.animation.rotations = Quaternions.from_euler(
                x[:,:nj*3].reshape((nf, nj, 3)), order='xyz', world=True)
                
            if self.translate:
                self.animation.positions = x[:,nj*3:].reshape((nf,nj, 3))

            if not self.silent:
                gp = Animation.positions_global(self.animation)
                gp = gp[:,np.array(list(self.targets.keys()))]
                error = np.mean(np.sum((self.endeff - gp)**2.0, axis=2)**0.5)
                print('[IK] Iteration %i | Error: %f' % (i+1, error))
            

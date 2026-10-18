import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import http from 'node:http';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { checkSdkProject, checkSdkPackage, writeSdkEnvSnippet, checkSdkApi } from '../src/sdk-checks.js';
const root = mkdtempSync(join(tmpdir(),'xcelsior-sdk-contract-'));
const oldCwd = process.cwd();
let grants=0,requests=0;
const server=http.createServer(async(req,res)=>{
  if(req.url==='/oauth/token'){
    let body='';for await(const part of req) body+=part;
    const params=new URLSearchParams(body);
    if(params.get('client_id')!=='test-client'||params.get('client_secret')!=='test-secret'){res.writeHead(401).end('{}');return;}
    grants++;
    res.setHeader('Content-Type','application/json');res.end(JSON.stringify({access_token:'runtime-'+grants,expires_in:0.2,token_type:'Bearer'}));return;
  }
  if(req.url==='/instances'&&req.headers.authorization?.startsWith('Bearer runtime-')){
    requests++;res.setHeader('Content-Type','application/json');res.end(JSON.stringify({ok:true,instances:[]}));return;
  }
  res.writeHead(404).end('{}');
});
try {
  writeFileSync(join(root,'package.json'),JSON.stringify({name:'xcelsior-sdk-contract',private:true,type:'module'}));
  writeFileSync(join(root,'.env'),'# existing\nUNRELATED_SETTING=keep\n');
  process.chdir(root);
  const project=checkSdkProject();if(project.some(r=>!r.ok))throw new Error(JSON.stringify(project));
  const installation=await checkSdkPackage();if(installation.some(r=>!r.ok))throw new Error(JSON.stringify(installation));
  await new Promise<void>(r=>server.listen(0,'127.0.0.1',r));
  const address=server.address() as {port:number};const base='http://127.0.0.1:'+address.port;
  const env=writeSdkEnvSnippet(base,'xoa_sign_in_not_to_persist','test-client','test-secret');
  if(readFileSync(env,'utf8').includes('xoa_sign_in'))throw new Error('Sign-in token persisted');
  if(!readFileSync(env,'utf8').includes('UNRELATED_SETTING=keep'))throw new Error('Existing environment lost');
  const verification=await checkSdkApi(base,'xoa_sign_in_not_to_persist','test-client','test-secret');
  if(verification.some(r=>!r.ok))throw new Error(JSON.stringify(verification));
  const before=grants;
  await promisify(execFile)(process.execPath,['--input-type=module','--eval',`const {client}=await import('./xcelsior-client.mjs');await Promise.all(Array.from({length:5},()=>client.instances.list()));await new Promise(r=>setTimeout(r,250));await client.instances.list();`],{cwd:root,timeout:30000});
  if(grants-before!==2)throw new Error('Expected one coalesced grant and one renewal, got '+(grants-before));
  console.log(JSON.stringify({installation,verification,oauth_grants:grants,instance_requests:requests,concurrent_and_renewed:true}));
} finally {process.chdir(oldCwd);server.close();rmSync(root,{recursive:true,force:true});}
